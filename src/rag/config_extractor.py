"""
Config extraction layer: text → ConfigDict (deterministic, RAG-based).

Pipeline:
    preprocess_text(text)
        → _extract_with_llm(text) | _extract_rule_based(text)
        → normalize_config(raw)        [from normalizer.py]
        → return ConfigDict
"""

import json
import re
from typing import Any, Dict, List, Tuple

from src.agents.types import ConfigDict
from src.rag.normalizer import normalize_config

try:
    from src.llm_client import llm_complete
    _HAS_LLM = True
except (ImportError, RuntimeError):
    _HAS_LLM = False


# ---------------------------------------------------------------------------
# Layer keyword patterns (order-preserving, applied to preprocessed text)
# Patterns ordered: longest/most-specific first to avoid partial matches
# ---------------------------------------------------------------------------

_LAYER_PATTERNS: List[Tuple[str, str]] = [
    # Multi-word patterns first (highest specificity)
    (r"\bmulti[_\s]?head[_\s]?(?:self[_\s]?)?attention\b", "multiheadattention"),
    (r"\bself[_\s]?attention\b",                           "multiheadattention"),
    # Transformer encoder/decoder semantically describe attention mechanisms
    (r"\btransformer[_\s]?encoder\b",                      "transformerencoder"),
    (r"\btransformer[_\s]?decoder\b",                      "transformerdecoder"),
    (r"\bfully[_\s]?connected\b",                          "linear"),
    (r"\bfully[_\s]?-[_\s]?connected\b",                   "linear"),
    (r"\bresidual[_\s]?block\b",                           "residualblock"),
    (r"\baverage[_\s]?pool(?:ing)?\b",                     "avgpool2d"),
    (r"\bavg[_\s]?pool(?:ing)?\b",                         "avgpool2d"),
    (r"\bmax[_\s]?pool(?:ing)?\b",                         "maxpool2d"),
    (r"\bbatch[_\s]?norm(?:alization)?\b",                "batchnorm2d"),
    (r"\blayer[_\s]?norm(?:alization)?\b",                "layernorm"),
    (r"\bconv(?:olution(?:al)?)?[_\s]?layer\b",           "conv2d"),

    # Single-word patterns (lower specificity)
    # Conv variants: conv, conv1d, conv2d, conv3d (with optional separators)
    (r"\bconv[1-3]?d?\b",                                 "conv2d"),
    (r"\btransformer\b",                      "transformerblock"),
    (r"\battention\b",                        "multiheadattention"),
    (r"\bfc\b",                               "linear"),   # Must come before other conv patterns
    (r"\bconv(?:olution(?:al)?)?\b",          "conv2d"),
    (r"\blinear\b",                           "linear"),
    (r"\bdense\b",                            "linear"),
    (r"\bupsamp(?:le|ling)\b",                "upsample"),
    (r"\bresiual\b",                          "residualblock"),   # typo
    (r"\bresidual\b",                         "residualblock"),
    (r"\brelu\b",                             "relu"),
    (r"\bdropout\b",                          "dropout"),
    (r"\bmha\b",                              "multiheadattention"), # abbreviation
    (r"\bmhsa\b",                             "multiheadattention"), # abbreviation
]

# Words that signal uncertain/approximate values — suppress param extraction.
_UNCERTAINTY_WORDS = re.compile(
    r"\b(maybe|perhaps|around|approximately|roughly|about|possibly|likely|"
    r"some|several|a few|typically|usually|often|sometimes)\b",
    re.IGNORECASE,
)

# Param extraction patterns: (canonical_key, [patterns_to_try])
_PARAM_PATTERNS: List[Tuple[str, List[str]]] = [
    ("kernel_size", [
        r"\b(\d+)\s*[x×]\s*\d+\s+(?:conv|kernel|filter)",
        r"(?:kernel|filter)[_\s]?size[:\s]+(\d+)",
        r"(\d+)[x×]\d+\s+kernel",
    ]),
    ("channels", [
        r"(\d+)\s+channels?",
        r"(\d+)\s+filters?",
        r"channels?\s*[:\=]\s*(\d+)",
        r"filters?\s*[:\=]\s*(\d+)",
    ]),
    ("stride", [
        r"strides?\s*[:\=]\s*(\d+)",
        r"stride\s+of\s+(\d+)",
    ]),
    ("padding", [
        r"padding\s*[:\=]\s*(\d+)",
    ]),
    ("hidden_size", [
        r"(\d+)\s+(?:hidden\s+)?units?",
        r"hidden[_\s]?(?:size|dim(?:ension)?)\s*[:\=]\s*(\d+)",
    ]),
]


class ConfigExtractor:
    """
    Extract architecture config from raw text.

    Always normalizes via normalize_config() before returning,
    ensuring identical schema regardless of extraction path.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and _HAS_LLM

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_text(self, text: str) -> ConfigDict:
        """
        text → ConfigDict (deterministic, validated).

        Flow:
            preprocess → (llm | rules) → normalize_config
        """
        processed = preprocess_text(text)

        try:
            if self.use_llm:
                raw = self._extract_with_llm(processed)
            else:
                raw = self._extract_rule_based(processed)
        except Exception:
            raw = self._extract_rule_based(processed)

        return normalize_config(raw)

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    def _extract_with_llm(self, text: str) -> Dict[str, Any]:
        """Call LLM with temperature=0, return raw dict."""
        prompt = (
            "Extract the neural network architecture from the text below.\n"
            "Return ONLY valid JSON — no explanation, no markdown.\n\n"
            "Required format:\n"
            '{"name": "ModelName", "layers": [{"type": "conv2d", "params": {"kernel_size": 3, "channels": 64}}]}\n\n'
            "Rules:\n"
            "- type must be one of: conv2d, conv1d, linear, maxpool2d, avgpool2d,\n"
            "  multiheadattention, transformerblock, batchnorm2d, layernorm, relu,\n"
            "  dropout, upsample, residualblock\n"
            "- params: only extract values EXPLICITLY stated in text\n"
            "- Do NOT infer, guess, or add default values\n\n"
            f"Text:\n{text}"
        )

        response = llm_complete(prompt)

        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Strip markdown code block
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        raise ValueError("LLM did not return valid JSON")

    # ------------------------------------------------------------------
    # Rule-based path
    # ------------------------------------------------------------------

    def _extract_rule_based(self, text: str) -> Dict[str, Any]:
        """Rule-based extraction producing same schema as LLM path."""
        return {
            "name": _extract_name(text),
            "layers": _extract_layers(text),
        }


# ---------------------------------------------------------------------------
# Module-level helpers (stateless, reusable)
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Normalize raw text for consistent layer detection.

    - Lowercase
    - Collapse whitespace
    - Normalize arrow variants (→, ->, =>, -->) to " then "
    - Normalize list punctuation (commas between operations → " then ")
    """
    text = text.lower()

    # Normalize arrows
    text = re.sub(r"\s*(?:->|=>|-->|[→⟶])\s*", " then ", text)

    # Collapse whitespace (preserve single newlines as sentence boundaries)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = text.strip()

    return text


def _extract_name(text: str) -> str:
    """
    Extract architecture name from preprocessed text (explicit only).

    Only matches when the name is preceded by a colon or an explicit
    label like "called", "named", "model:", etc.
    Avoids grabbing verbs like "starts", "uses", etc.
    """
    patterns = [
        r"^([a-z][a-z0-9\-]+)\s*:",                        # "ResNet-18: ..."
        r"(?:called|named|model\s*:)\s*([a-z][a-z0-9\-]+)", # "named ResNet"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Reject common verbs / prepositions that can follow "model"
            _stop = {"the", "a", "an", "this", "that", "with", "from",
                     "starts", "uses", "has", "contains", "consists"}
            if name.lower() not in _stop and len(name) > 1:
                return name

    return "UnknownModel"


def _extract_layers(text: str) -> List[Dict[str, Any]]:
    """
    Detect layers in ORDER of appearance in text.

    Handles:
    - Single layers (e.g., "conv2d")
    - Multi-block phrases (e.g., "3 residual blocks")

    Preserves position-based ordering.

    IMPORTANT: Layer ordering is TEXTUAL (position-based), not semantic.
    The order reflects the order of keywords in the input text, not any
    semantic or architectural ordering. Users should provide text in the
    order they want layers to appear in the final architecture.
    """
    # Step 1: Detect multi-block phrases (e.g., "3 residual blocks")
    multi_hits = _detect_multi_blocks(text)

    # Step 2: Detect single layers (convert multi_hits for _pos_claimed)
    single_hits: List[Tuple[int, str]] = []
    for pattern, canonical_type in _LAYER_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            # Convert multi_hits tuples to (pos, type) for comparison
            multi_hits_2d = [(pos, typ) for pos, typ, _ in multi_hits]
            if not _pos_claimed(m.start(), single_hits + multi_hits_2d):
                single_hits.append((m.start(), canonical_type))

    # Step 3: Merge and sort by position
    # Expand multi_hits to (pos, type) format for merging
    all_hits: List[Tuple[int, str, int]] = []
    for pos, typ, count in multi_hits:
        all_hits.append((pos, typ, count))
    for pos, typ in single_hits:
        all_hits.append((pos, typ, 1))  # Single hits have count=1 (no expansion)

    # Sort by position first, then by type for deterministic ordering
    # When positions are equal, consistent type ordering ensures stability
    all_hits.sort(key=lambda h: (h[0], h[1]))

    # Step 4: Build repeat groups (track which layers belong to same multi-block expansion)
    repeat_groups: Dict[int, List[int]] = {}  # pos -> list of indices in all_hits
    for idx, (pos, _, _) in enumerate(all_hits):
        if pos not in repeat_groups:
            repeat_groups[pos] = []
        repeat_groups[pos].append(idx)

    layers = []
    for idx, (pos, layer_type, original_count) in enumerate(all_hits):
        params = _extract_params_near(text, pos)

        # Add repeat metadata for multi-block expansions
        group_indices = repeat_groups[pos]
        if len(group_indices) > 1:
            # This is part of a multi-block expansion
            local_index = group_indices.index(idx)
            actual_count = len(group_indices)

            # Use single underscore for internal metadata (not double)
            # Smart naming: add "_block" only if type doesn't already end with "block"
            repeat_group = (
                layer_type
                if layer_type.endswith("block")
                else f"{layer_type}_block"
            )
            params["_repeat_group"] = repeat_group  # Semantic group identifier
            params["_repeat_index"] = local_index  # 0-based index within group
            params["_repeat_total"] = actual_count  # Total in group

            # Flag if this expansion was truncated (original > 10)
            if original_count > 10:
                params["_repeat_truncated"] = True

        layers.append({"type": layer_type, "params": params})

    if not layers:
        layers = [{"type": "conv2d", "params": {}}]

    return layers


def _detect_multi_blocks(text: str) -> List[Tuple[int, str, int]]:
    """
    Detect patterns like "3 residual blocks" or "2 conv layers".

    Returns list of (pos, type, count) tuples, with count indicating how many
    expansions are requested (may be > 10 if capped).

    Examples:
    - "3 residual blocks" → 3 x (pos, "residualblock", 3)
    - "2 conv layers" → 2 x (pos, "conv2d", 2)
    - "15 residual blocks" → 10 x (pos, "residualblock", 15) [capped, but count=15 tracks original]

    IMPORTANT: Prevents overlapping phrase matches by tracking matched ranges.
    If two multi-block phrases overlap in the text (e.g., "3 conv layers blocks"),
    only the first match is kept.

    NOTE:
    Repeat metadata (_repeat_group, _repeat_index, _repeat_total)
    preserves structural patterns for future reasoning and compression.
    Current system treats all nodes as flat (no composite expansion).
    """
    hits: List[Tuple[int, str, int]] = []
    matched_ranges: List[Tuple[int, int]] = []  # Track (start, end) of matched phrases

    # Pattern: digit(s) + optional word + layer type + optional "s" for plural
    # E.g., "3 residual blocks", "2 conv layers", "4 dense layers"
    pattern = r"(\d+)\s+(?:(?:conv|residual|dense|linear|attention|transformer)\w*\s+)?(blocks?|layers?)"

    for m in re.finditer(pattern, text, re.IGNORECASE):
        count = int(m.group(1))
        block_phrase = m.group(0)
        phrase_start = m.start()
        phrase_end = m.end()

        # Check if this range overlaps with any already matched range
        overlaps = any(
            not (phrase_end <= start or phrase_start >= end)
            for start, end in matched_ranges
        )
        if overlaps:
            continue  # Skip overlapping matches

        # Detect the layer type mentioned in this phrase
        layer_type = None
        for kw_pattern, canonical in _LAYER_PATTERNS:
            if re.search(kw_pattern, block_phrase, re.IGNORECASE):
                layer_type = canonical
                break

        if layer_type:
            # Record this matched range to prevent overlaps
            matched_ranges.append((phrase_start, phrase_end))

            # Cap expansion at 10 for practical limits
            actual_count = min(count, 10)

            # Add `actual_count` copies of this layer at the same position
            # (they'll be expanded to separate nodes but keep relative ordering)
            # Include original count so we can track if truncation occurred
            for _ in range(actual_count):
                hits.append((phrase_start, layer_type, count))

    return hits


def _pos_claimed(pos: int, hits: List[Tuple[int, str]], tol: int = 10) -> bool:
    """Check if a character position is already covered by a recorded hit."""
    return any(abs(pos - h[0]) < tol for h in hits)


def _extract_params_near(text: str, pos: int) -> Dict[str, Any]:
    """
    Extract explicit parameters in the window around `pos`.

    Window is tight (look-back 20 chars, look-ahead 80 chars) to
    avoid bleeding parameters from adjacent layer descriptions.
    Strict: if an uncertainty word appears near the value, skip it.
    """
    window_start = max(0, pos - 20)
    window_end   = min(len(text), pos + 80)
    window = text[window_start:window_end]

    params: Dict[str, Any] = {}

    for param_key, patterns in _PARAM_PATTERNS:
        for pattern in patterns:
            match = re.search(pattern, window, re.IGNORECASE)
            if not match:
                continue

            # Reject if uncertainty word precedes the value in the window
            pre_context = window[: match.start()]
            # Only check the last 30 chars before the match
            near_pre = pre_context[-30:]
            if _UNCERTAINTY_WORDS.search(near_pre):
                continue

            try:
                params[param_key] = int(match.group(1))
                break   # First matching pattern wins; no double-extraction
            except (ValueError, IndexError):
                continue

    return params
