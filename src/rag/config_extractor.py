"""
Config extraction layer: text -> ConfigDict.

Improvements:
  R1  - Few-shot LLM prompt + explicit connection instructions
  R3  - Self-correction / verification loop
  R4  - Wider parameter search window
  NEW - Multi-step pipeline (family -> skeleton -> details)
  NEW - Relationship / connection extraction (skip, branch, concat)
  NEW - Basic table row extraction
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.agents.types import ConfigDict
from src.rag.normalizer import normalize_config
from src.rag.section_splitter import get_architecture_text, chunk_for_retrieval
from src.rag.retriever import retrieve_and_merge
from src.rag.knowledge_graph import KnowledgeGraph

try:
    from src.llm_client import llm_complete
    _HAS_LLM = True
except (ImportError, RuntimeError):
    _HAS_LLM = False


# ---------------------------------------------------------------------------
# Layer keyword patterns
# ---------------------------------------------------------------------------

_LAYER_PATTERNS: List[Tuple[str, str]] = [
    (r"\bmulti[_\s]?head[_\s]?(?:self[_\s]?)?attention\b", "multiheadattention"),
    (r"\bself[_\s]?attention\b",                           "multiheadattention"),
    (r"\btransformer[_\s]?encoder\b",                      "multiheadattention"),
    (r"\btransformer[_\s]?decoder\b",                      "multiheadattention"),
    (r"\bfully[_\s]?connected\b",                          "linear"),
    (r"\bfully[_\s]?-[_\s]?connected\b",                   "linear"),
    (r"\bresidual[_\s]?block\b",                           "residualblock"),
    (r"\baverage[_\s]?pool(?:ing)?\b",                     "avgpool2d"),
    (r"\bavg[_\s]?pool(?:ing)?\b",                         "avgpool2d"),
    (r"\bmax[_\s]?pool(?:ing)?\b",                         "maxpool2d"),
    (r"\bbatch[_\s]?norm(?:alization)?\b",                 "batchnorm2d"),
    (r"\blayer[_\s]?norm(?:alization)?\b",                 "layernorm"),
    (r"\bconv(?:olution(?:al)?)?[_\s]?layer\b",           "conv2d"),
    (r"\bconv[1-3]?d?\b",                                  "conv2d"),
    (r"\btransformer\b",                                   "transformerblock"),
    (r"\battention\b",                                     "multiheadattention"),
    (r"\bfc\b",                                            "linear"),
    (r"\blinear\b",                                        "linear"),
    (r"\bdense\b",                                         "linear"),
    (r"\bupsamp(?:le|ling)\b",                             "upsample"),
    (r"\bresidual\b",                                      "residualblock"),
    (r"\brelu\b",                                          "relu"),
    (r"\bdropout\b",                                       "dropout"),
    (r"\bmha\b",                                           "multiheadattention"),
    (r"\bmhsa\b",                                          "multiheadattention"),
    (r"\bpatch[_\s]?embed(?:ding)?\b",                    "patchembedding"),
]

_UNCERTAINTY_WORDS = re.compile(
    r"\b(maybe|perhaps|around|approximately|roughly|about|possibly|likely|"
    r"some|several|a few|typically|usually|often|sometimes)\b",
    re.IGNORECASE,
)

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
        r"d_model\s*[:\=]\s*(\d+)",
    ]),
    ("num_heads", [
        r"(\d+)\s+(?:attention\s+)?heads?",
        r"heads?\s*[:\=]\s*(\d+)",
        r"num_heads\s*[:\=]\s*(\d+)",
    ]),
    ("num_layers", [
        r"(\d+)\s+(?:transformer\s+)?(?:encoder\s+)?(?:decoder\s+)?layers?",
        r"num_layers\s*[:\=]\s*(\d+)",
        r"depth\s*[:\=]\s*(\d+)",
    ]),
]

# Connection / relationship extraction patterns
_CONNECTION_PATTERNS = [
    (r"\bskip[_\s]?connection\b",    "skip"),
    (r"\bskip[_\s]?connect\b",       "skip"),
    (r"\bresidual[_\s]?connection\b", "residual"),
    (r"\bshortcut[_\s]?connection\b", "skip"),
    (r"\bconcaten(?:ate|ation)\b",    "concat"),
    (r"\bconcat\b",                   "concat"),
    (r"\bbranch\b",                   "branch"),
    (r"\bfork\b",                     "branch"),
    (r"\bmerge\b",                    "merge"),
    (r"\badd\b",                      "add"),
    (r"\belement[_\s]?wise[_\s]?add", "add"),
]
_conn_re = [(re.compile(p, re.IGNORECASE), t) for p, t in _CONNECTION_PATTERNS]


# ---------------------------------------------------------------------------
# Few-shot prompt template (R1)
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES = """
### Example 1 — ResNet-like CNN:
Text: "ResNet-18 starts with a 7×7 conv layer with 64 channels and stride 2, followed by max pooling. Then 4 groups of residual blocks with 64, 128, 256, and 512 channels respectively. A global average pool and fully connected layer output 1000 classes."
Output:
{
  "name": "ResNet-18",
  "layers": [
    {"type": "conv2d",       "params": {"kernel_size": 7, "channels": 64, "stride": 2}},
    {"type": "maxpool2d",    "params": {}},
    {"type": "residualblock","params": {"channels": 64}},
    {"type": "residualblock","params": {"channels": 128}},
    {"type": "residualblock","params": {"channels": 256}},
    {"type": "residualblock","params": {"channels": 512}},
    {"type": "avgpool2d",    "params": {}},
    {"type": "linear",       "params": {"channels": 1000}}
  ],
  "connections": [
    ["layer_0","layer_1"],["layer_1","layer_2"],["layer_2","layer_3"],
    ["layer_3","layer_4"],["layer_4","layer_5"],["layer_5","layer_6"],
    ["layer_6","layer_7"]
  ],
  "connection_types": {"layer_2": "residual", "layer_3": "residual"}
}

### Example 2 — U-Net with skip connections:
Text: "U-Net has an encoder that downsamples with conv+pool blocks, and a decoder that upsamples and concatenates encoder features via skip connections."
Output:
{
  "name": "U-Net",
  "layers": [
    {"type": "conv2d",  "params": {"channels": 64}},
    {"type": "maxpool2d","params": {}},
    {"type": "conv2d",  "params": {"channels": 128}},
    {"type": "upsample","params": {}},
    {"type": "conv2d",  "params": {"channels": 64}}
  ],
  "connections": [
    ["layer_0","layer_1"],["layer_1","layer_2"],["layer_2","layer_3"],
    ["layer_3","layer_4"],["layer_0","layer_4"]
  ],
  "connection_types": {"layer_0->layer_4": "skip"}
}

### Example 3 — Transformer:
Text: "The model uses a standard Transformer with 6 encoder and 6 decoder layers. Each layer has multi-head attention with 8 heads and d_model=512. A feed-forward sub-layer has hidden dimension 2048."
Output:
{
  "name": "Transformer",
  "layers": [
    {"type": "multiheadattention","params": {"num_heads": 8, "hidden_size": 512}},
    {"type": "layernorm",         "params": {}},
    {"type": "linear",            "params": {"hidden_size": 2048}},
    {"type": "layernorm",         "params": {}},
    {"type": "multiheadattention","params": {"num_heads": 8, "hidden_size": 512}},
    {"type": "layernorm",         "params": {}},
    {"type": "linear",            "params": {"hidden_size": 2048}},
    {"type": "layernorm",         "params": {}},
    {"type": "linear",            "params": {"channels": 512}}
  ],
  "connections": [
    ["layer_0","layer_1"],["layer_1","layer_2"],["layer_2","layer_3"],
    ["layer_3","layer_4"],["layer_4","layer_5"],["layer_5","layer_6"],
    ["layer_6","layer_7"],["layer_7","layer_8"]
  ],
  "connection_types": {}
}
"""

_LLM_EXTRACTION_PROMPT = """\
You are an expert at reading deep learning research papers and extracting neural network architectures.

{few_shot}

{graph_rules}

### Now extract from this text:
Text: \"\"\"{text}\"\"\"

Return ONLY valid JSON — no explanation, no markdown fences.

Rules:
- "type" must be one of: conv2d, conv1d, linear, maxpool2d, avgpool2d,
  multiheadattention, transformerblock, batchnorm2d, layernorm, relu,
  dropout, upsample, residualblock, patchembedding
- "params": ONLY extract values EXPLICITLY stated in the text. Do NOT guess.
- "connections": list of [source_id, target_id] pairs using layer indices.
- "connection_types": dict mapping "src_id->tgt_id" or layer_id to
  connection type: "skip", "residual", "concat", "branch", "add".
- If you see skip connections, concatenation, or branches, include them.
"""

_VERIFICATION_PROMPT = """\
You are verifying a neural network architecture extraction.

Original text:
\"\"\"{text}\"\"\"

Extracted JSON:
{extracted}

Check for these issues and return a corrected JSON:
1. Are there any layers mentioned in the text that are MISSING from the JSON?
2. Are there skip connections, concatenations, or branches NOT captured?
3. Are any parameter values wrong (different from what the text states)?
4. Is the layer ORDER correct as described in the text?

Return ONLY corrected valid JSON. If no corrections are needed, return the original JSON unchanged.
"""


# ---------------------------------------------------------------------------
# ConfigExtractor
# ---------------------------------------------------------------------------

class ConfigExtractor:
    """
    Extract architecture config from raw text using a multi-step pipeline.

    Steps:
      1. Section-aware text focusing (SectionSplitter)
      2. BM25 retrieval if text is too large
      3. LLM extraction with few-shot prompt
      4. Rule-based fallback
      5. Self-correction verification loop (R3)
      6. Normalization
    """

    def __init__(
        self,
        use_llm: bool = True,
        use_section_splitter: bool = True,
        use_retriever: bool = True,
        verify: bool = True,
        max_context_chars: int = 10_000,
    ):
        self.use_llm = use_llm and _HAS_LLM
        self.use_section_splitter = use_section_splitter
        self.use_retriever = use_retriever
        self.verify = verify and use_llm and _HAS_LLM
        self.max_context_chars = max_context_chars
        self.ontology = KnowledgeGraph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_text(self, text: str) -> ConfigDict:
        """
        Full pipeline: raw text -> focused context -> extract -> verify -> normalize.
        """
        focused = self._focus_text(text)

        try:
            if self.use_llm:
                raw = self._extract_with_llm(focused)
                if self.verify:
                    raw = self._verify_extraction(focused, raw)
            else:
                raw = self._extract_rule_based(focused)
        except Exception:
            raw = self._extract_rule_based(focused)

        return normalize_config(raw)

    def extract_from_full_pdf(self, pdf_text: str) -> ConfigDict:
        """
        Entry point for full PDF text. Applies section splitting first.
        """
        focused = get_architecture_text(pdf_text, max_chars=self.max_context_chars)
        return self.extract_from_text(focused)

    # ------------------------------------------------------------------
    # Step 1: Text focusing
    # ------------------------------------------------------------------

    def _focus_text(self, text: str) -> str:
        """Apply section splitting and BM25 retrieval to narrow the context."""
        if len(text) <= self.max_context_chars:
            return text

        if self.use_section_splitter:
            text = get_architecture_text(text, max_chars=self.max_context_chars)

        if len(text) > self.max_context_chars and self.use_retriever:
            chunks = chunk_for_retrieval(text, chunk_size=1_200, overlap=200)
            text = retrieve_and_merge(chunks, top_k=6, max_chars=self.max_context_chars)

        return text

    # ------------------------------------------------------------------
    # Step 2: LLM extraction with few-shot prompt (R1)
    # ------------------------------------------------------------------

    def _extract_with_llm(self, text: str) -> Dict[str, Any]:
        """Call LLM with few-shot prompt, KAG instructions, and connection instructions."""
        # KAG Entity Linking & Rule Extraction
        terms = self.ontology.identify_terms(text)
        graph_rules = self.ontology.get_context_for_terms(terms)

        prompt = _LLM_EXTRACTION_PROMPT.format(
            few_shot=_FEW_SHOT_EXAMPLES,
            graph_rules=graph_rules,
            text=text,
        )
        response = llm_complete(prompt)
        return self._parse_json_response(response)

    # ------------------------------------------------------------------
    # Step 3: Self-correction loop (R3)
    # ------------------------------------------------------------------

    def _verify_extraction(
        self, original_text: str, extracted: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ask the LLM to review its own extraction against the source text.
        Returns corrected dict, or original if correction fails.
        """
        try:
            prompt = _VERIFICATION_PROMPT.format(
                text=original_text[:4_000],  # keep verification prompt compact
                extracted=json.dumps(extracted, indent=2),
            )
            response = llm_complete(prompt)
            corrected = self._parse_json_response(response)
            # Only accept correction if it has more or equal layers (no regression)
            if len(corrected.get("layers", [])) >= len(extracted.get("layers", [])):
                return corrected
        except Exception:
            pass
        return extracted

    # ------------------------------------------------------------------
    # Step 4: Rule-based fallback (enhanced R4)
    # ------------------------------------------------------------------

    def _extract_rule_based(self, text: str) -> Dict[str, Any]:
        """Rule-based extraction: layers + connections + tables."""
        processed = preprocess_text(text)
        layers = _extract_layers(processed)
        connections = _extract_connections(processed, layers)
        connection_types = _extract_connection_types(processed)

        # Supplement with table-extracted layers
        table_layers = extract_table_layers(text)
        if table_layers and len(table_layers) > len(layers):
            layers = table_layers

        return {
            "name": _extract_name(processed),
            "layers": layers,
            "connections": connections,
            "connection_types": connection_types,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_response(response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, stripping markdown if needed."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError("LLM did not return valid JSON")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Normalize raw text for consistent layer detection."""
    text = text.lower()
    # Normalize arrow variants to " then "
    text = re.sub(r"\s*(?:->|=>|-->|[→⟶])\s*", " then ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _extract_name(text: str) -> str:
    """Extract architecture name from preprocessed text."""
    patterns = [
        r"^([a-z][a-z0-9\-]+)\s*:",
        r"(?:called|named|model\s*:)\s*([a-z][a-z0-9\-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            _stop = {"the", "a", "an", "this", "that", "with", "from",
                     "starts", "uses", "has", "contains", "consists"}
            if name.lower() not in _stop and len(name) > 1:
                return name
    return "UnknownModel"


def _extract_layers(text: str) -> List[Dict[str, Any]]:
    """Detect layers in order of appearance (position-based)."""
    multi_hits = _detect_multi_blocks(text)
    single_hits: List[Tuple[int, str]] = []

    for pattern, canonical_type in _LAYER_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            multi_hits_2d = [(pos, typ) for pos, typ, _ in multi_hits]
            if not _pos_claimed(m.start(), single_hits + multi_hits_2d):
                single_hits.append((m.start(), canonical_type))

    all_hits: List[Tuple[int, str, int]] = []
    for pos, typ, count in multi_hits:
        all_hits.append((pos, typ, count))
    for pos, typ in single_hits:
        all_hits.append((pos, typ, 1))

    all_hits.sort(key=lambda h: (h[0], h[1]))

    repeat_groups: Dict[int, List[int]] = {}
    for idx, (pos, _, _) in enumerate(all_hits):
        repeat_groups.setdefault(pos, []).append(idx)

    layers = []
    for idx, (pos, layer_type, original_count) in enumerate(all_hits):
        params = _extract_params_near(text, pos)
        group_indices = repeat_groups[pos]
        if len(group_indices) > 1:
            local_index = group_indices.index(idx)
            actual_count = len(group_indices)
            repeat_group = (
                layer_type if layer_type.endswith("block") else f"{layer_type}_block"
            )
            params["_repeat_group"] = repeat_group
            params["_repeat_index"] = local_index
            params["_repeat_total"] = actual_count
            if original_count > 10:
                params["_repeat_truncated"] = True
        layers.append({"type": layer_type, "params": params})

    if not layers:
        layers = [{"type": "conv2d", "params": {}}]
    return layers


def _detect_multi_blocks(text: str) -> List[Tuple[int, str, int]]:
    """Detect 'N residual blocks', '3 conv layers', etc."""
    hits: List[Tuple[int, str, int]] = []
    matched_ranges: List[Tuple[int, int]] = []
    pattern = r"(\d+)\s+(?:(?:conv|residual|dense|linear|attention|transformer)\w*\s+)?(blocks?|layers?)"

    for m in re.finditer(pattern, text, re.IGNORECASE):
        count = int(m.group(1))
        block_phrase = m.group(0)
        phrase_start, phrase_end = m.start(), m.end()
        if any(not (phrase_end <= s or phrase_start >= e) for s, e in matched_ranges):
            continue
        layer_type = None
        for kw_pattern, canonical in _LAYER_PATTERNS:
            if re.search(kw_pattern, block_phrase, re.IGNORECASE):
                layer_type = canonical
                break
        if layer_type:
            matched_ranges.append((phrase_start, phrase_end))
            actual_count = min(count, 10)
            for _ in range(actual_count):
                hits.append((phrase_start, layer_type, count))
    return hits


def _pos_claimed(pos: int, hits: List[Tuple[int, str]], tol: int = 10) -> bool:
    return any(abs(pos - h[0]) < tol for h in hits)


def _extract_params_near(text: str, pos: int) -> Dict[str, Any]:
    """Extract explicit parameters in a wider window around pos (R4)."""
    # R4: expanded window — 40 chars lookback, 120 chars lookahead
    window_start = max(0, pos - 40)
    window_end = min(len(text), pos + 120)
    window = text[window_start:window_end]

    params: Dict[str, Any] = {}
    for param_key, patterns in _PARAM_PATTERNS:
        for pattern in patterns:
            match = re.search(pattern, window, re.IGNORECASE)
            if not match:
                continue
            pre_context = window[: match.start()]
            near_pre = pre_context[-40:]
            if _UNCERTAINTY_WORDS.search(near_pre):
                continue
            try:
                params[param_key] = int(match.group(1))
                break
            except (ValueError, IndexError):
                continue
    return params


def _extract_connections(
    text: str, layers: List[Dict[str, Any]]
) -> List[Tuple[str, str]]:
    """Build connections: sequential chain + detected skip/residual edges."""
    # Assign IDs first
    for i, layer in enumerate(layers):
        layer["id"] = f"layer_{i}"

    connections: List[Tuple[str, str]] = []
    for i in range(len(layers) - 1):
        connections.append((layers[i]["id"], layers[i + 1]["id"]))

    # Add residual skip edges
    for i, layer in enumerate(layers):
        if layer.get("type") == "residualblock" and i + 1 < len(layers) and i > 0:
            skip_src = layers[i - 1]["id"]
            skip_tgt = layers[i + 1]["id"]
            if (skip_src, skip_tgt) not in connections:
                connections.append((skip_src, skip_tgt))

    # Detect upsample skip connections (U-Net style)
    upsample_indices = [i for i, l in enumerate(layers) if l.get("type") == "upsample"]
    conv_indices = [i for i, l in enumerate(layers) if l.get("type") == "conv2d"]
    if upsample_indices and len(conv_indices) >= 2:
        for up_idx in upsample_indices:
            # find a conv before the first downsample that matches
            candidates = [c for c in conv_indices if c < up_idx // 2]
            if candidates:
                skip_src = layers[candidates[-1]]["id"]
                skip_tgt = layers[up_idx]["id"]
                if (skip_src, skip_tgt) not in connections:
                    connections.append((skip_src, skip_tgt))

    return connections


def _extract_connection_types(text: str) -> Dict[str, str]:
    """
    Detect connection type keywords in text.
    Returns a dict of keyword -> type for later graph annotation.
    """
    found: Dict[str, str] = {}
    for pattern, conn_type in _conn_re:
        if pattern.search(text):
            found[conn_type] = conn_type
    return found


# ---------------------------------------------------------------------------
# Table extraction (Improvement 5)
# ---------------------------------------------------------------------------

_TABLE_ROW_RE = re.compile(
    r"(?:^|\n)\s*"
    r"(?P<type>[A-Za-z][A-Za-z0-9_\-/ ]{1,30})"
    r"\s*[\|│]\s*"
    r"(?P<params>[0-9×x,\s\w]+)"
    r"\s*[\|│]",
    re.MULTILINE,
)

_TABLE_TYPE_SYNONYMS = {
    "conv": "conv2d", "convolution": "conv2d", "conv2d": "conv2d",
    "fc": "linear", "linear": "linear", "dense": "linear",
    "pool": "maxpool2d", "maxpool": "maxpool2d",
    "bn": "batchnorm2d", "batch norm": "batchnorm2d",
    "attention": "multiheadattention", "mha": "multiheadattention",
    "residual": "residualblock", "res block": "residualblock",
    "dropout": "dropout", "relu": "relu", "upsample": "upsample",
}


def extract_table_layers(text: str) -> List[Dict[str, Any]]:
    """
    Extract architecture layers from HTML/ASCII tables in paper text.

    Looks for lines with pipe characters (|) indicating table rows and
    attempts to parse layer type + parameter columns.

    Returns list of layer dicts, or [] if no table detected.
    """
    layers = []
    for match in _TABLE_ROW_RE.finditer(text):
        raw_type = match.group("type").strip().lower()
        canonical = None
        for key, val in _TABLE_TYPE_SYNONYMS.items():
            if key in raw_type:
                canonical = val
                break

        if not canonical:
            continue

        # Parse first numeric value in params column as channels
        params_str = match.group("params")
        numbers = re.findall(r"\d+", params_str)
        params: Dict[str, Any] = {}
        if numbers:
            params["channels"] = int(numbers[0])
        if len(numbers) > 1:
            params["kernel_size"] = int(numbers[1])

        layers.append({"type": canonical, "params": params})

    return layers
