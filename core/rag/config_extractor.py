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
import logging
import os
import re
from collections.abc import Callable
from typing import Any

from core.agents.types import ConfigDict
from core.knowledge.operations import OPERATIONS, find_mentioned
from core.rag.knowledge_graph import KnowledgeGraph
from core.rag.normalizer import normalize_config
from core.rag.retriever import retrieve_and_merge, retrieve_top_chunks
from core.rag.section_splitter import chunk_for_retrieval, get_architecture_text

logger = logging.getLogger(__name__)

# Natural-language stand-in for the architecture-focused BM25 query terms,
# used when ranking real (page/section-aware) chunks via a chunk_retriever
# callback -- dense retrieval scores natural language, not discrete terms.
_ARCHITECTURE_QUERY = (
    "convolution attention transformer residual connections layers "
    "channels kernel stride architecture design encoder decoder block "
    # Activation, normalisation and pooling vocabulary. Their absence was the
    # same failure as the missing numerics found in Phase 6: EfficientNet
    # states "We also use SiLU (Swish-1) activation" in prose that the query
    # gave the retriever no reason to rank, so the paper's only evidence for
    # an expected layer type never reached the model.
    "activation relu gelu silu swish sigmoid "
    "normalization batchnorm layernorm pooling dropout embedding "
    "64 128 256 512 768 1024 2048"
)


# Focused-context composition. Tables and captions carry the hyperparameters
# but lose to prose on an architecture-vocabulary query, so a bounded number
# of slots is reserved for them rather than left to ranking.
def _bare_model_id(model: str | None) -> str:
    """Strip a litellm routing prefix: 'groq/openai/gpt-oss-120b' -> 'gpt-oss-120b'."""
    return str(model or "").rsplit("/", 1)[-1].strip().lower()


# Reasoning effort for the extraction and verification calls. gpt-oss-120b's
# reasoning trace diverges between identical calls at temperature=0, and the
# extracted layer list diverges with it (3 vs 14 layers for one prompt).
# "low" was the only setting that returned identical completions for
# identical input, which is what makes the benchmark measurable at all.
_EXTRACTION_REASONING_EFFORT = os.getenv("EXTRACTION_REASONING_EFFORT", "low").strip() or None

_TOTAL_FOCUS_SLOTS = 6
_RESERVED_STRUCTURED_SLOTS = 2
_STRUCTURED_CHUNK_TYPES = ("table", "caption")
_FOCUS_SEPARATOR = "\n\n---\n\n"
_FORWARD_EXPANSION_DEPTH = 2
_STRUCTURED_LABEL_PREFIX = re.compile(
    r"^\s*(?:table|figure|fig\.?)\s*\d+\s*[:.)]?\s*", re.IGNORECASE
)
_ARCHITECTURAL_STRUCTURED_TERMS = re.compile(
    r"\b(?:conv(?:\d+(?:x|×)\d+)?|mbconv\d*|channels?|kernel|stride|layers?|"
    r"hidden|heads?|block|encoder|decoder|transformer|pooling|patch|embedding|"
    r"resolution|feature)\b",
    re.IGNORECASE,
)


def _has_architectural_structured_content(text: str) -> bool:
    """Whether a table/caption contains numeric architectural information."""
    content = _STRUCTURED_LABEL_PREFIX.sub("", text, count=1)
    return bool(re.search(r"\d", content) and _ARCHITECTURAL_STRUCTURED_TERMS.search(content))


try:
    from core.llm_client import PRIMARY_MODEL, get_last_completion_model, llm_complete

    _HAS_LLM = True
except (ImportError, RuntimeError):
    _HAS_LLM = False


# ---------------------------------------------------------------------------
# Layer keyword patterns
# ---------------------------------------------------------------------------

_LAYER_PATTERNS: list[tuple[str, str]] = [
    (r"\bdepthwise(?:[_\s-]+separable)?[_\s-]+conv(?:olution)?\b", "depthwise_conv2d"),
    (
        r"\b(?:(?:transposed|transpose|de)[_\s-]*convolutions?|"
        r"fractional(?:ly)?[_\s-]*strided[_\s-]*convolutions?)\b",
        "convtranspose2d",
    ),
    (r"\bconcaten(?:ate|ation)\b", "concat"),
    (r"\bpositional[_\s]?embed(?:ding)?\b", "positionalembedding"),
    (r"\bglobal[_\s]?(?:average|avg)[_\s]?pool(?:ing)?\b", "globalavgpool2d"),
    (r"\bfeed[_\s-]?forward\b", "feedforward"),
    (r"\bgroup[_\s]?norm(?:alization)?\b", "groupnorm"),
    (r"\bleaky[_\s]?relu\b", "leakyrelu"),
    (r"\b(?:silu|swish)\b", "silu"),
    (r"\bcls[_\s-]?token\b", "clstoken"),
    (r"\bsequence[_\s-]?pool(?:ing)?\b", "sequence_pooling"),
    (r"\bflatten(?:ing)?\b", "flatten"),
    (r"\bgelu\b", "gelu"),
    (r"\b(?:residual[_\s-]?add|element[_\s-]?wise[_\s-]?add)\b", "residual_add"),
    (r"\bmulti[_\s]?head[_\s]?(?:self[_\s]?)?attention\b", "multiheadattention"),
    (r"\bself[_\s]?attention\b", "multiheadattention"),
    (r"\btransformer[_\s]?encoder\b", "multiheadattention"),
    (r"\btransformer[_\s]?decoder\b", "multiheadattention"),
    (r"\bfully[_\s]?connected\b", "linear"),
    (r"\bfully[_\s]?-[_\s]?connected\b", "linear"),
    (r"\bresidual[_\s]?block\b", "residualblock"),
    (r"\baverage[_\s]?pool(?:ing)?\b", "avgpool2d"),
    (r"\bavg[_\s]?pool(?:ing)?\b", "avgpool2d"),
    (r"\bmax[_\s]?pool(?:ing)?\b", "maxpool2d"),
    (r"\bbatch[_\s]?norm(?:alization)?\b", "batchnorm2d"),
    (r"\blayer[_\s]?norm(?:alization)?\b", "layernorm"),
    (r"\bconv(?:olution(?:al)?)?[_\s]?layer\b", "conv2d"),
    (r"\bconv[1-3]?d?\b", "conv2d"),
    (r"\btransformer\b", "transformerblock"),
    (r"\battention\b", "multiheadattention"),
    (r"\bfc\b", "linear"),
    (r"\blinear\b", "linear"),
    (r"\bdense\b", "linear"),
    (r"\bupsamp(?:le|ling)\b", "upsample"),
    (r"\bresidual\b", "residualblock"),
    (r"\brelu\b", "relu"),
    (r"\bdropout\b", "dropout"),
    (r"\bmha\b", "multiheadattention"),
    (r"\bmhsa\b", "multiheadattention"),
    (r"\bpatch[_\s]?embed(?:ding)?\b", "patchembedding"),
]

_UNCERTAINTY_WORDS = re.compile(
    r"\b(maybe|perhaps|around|approximately|roughly|about|possibly|likely|"
    r"some|several|a few|typically|usually|often|sometimes)\b",
    re.IGNORECASE,
)

_PARAM_PATTERNS: list[tuple[str, list[str]]] = [
    (
        "kernel_size",
        [
            r"\b(\d+)\s*[x×]\s*\d+\s+(?:conv|kernel|filter)",
            r"(?:kernel|filter)[_\s]?size[:\s]+(\d+)",
            r"(\d+)[x×]\d+\s+kernel",
        ],
    ),
    (
        "channels",
        [
            r"(\d+)\s+channels?",
            r"(\d+)\s+filters?",
            r"channels?\s*[:\=]\s*(\d+)",
            r"filters?\s*[:\=]\s*(\d+)",
        ],
    ),
    (
        "stride",
        [
            r"strides?\s*[:\=]\s*(\d+)",
            r"stride\s+of\s+(\d+)",
            r"\bstride\s+(\d+)\b",
        ],
    ),
    (
        "padding",
        [
            r"padding\s*[:\=]\s*(\d+)",
        ],
    ),
    (
        "hidden_size",
        [
            r"(\d+)\s+(?:hidden\s+)?units?",
            r"hidden[_\s]?(?:size|dim(?:ension)?)\s*[:\=]\s*(\d+)",
            r"d_model\s*[:\=]\s*(\d+)",
        ],
    ),
    (
        "num_heads",
        [
            r"(\d+)\s+(?:attention\s+)?heads?",
            r"heads?\s*[:\=]\s*(\d+)",
            r"num_heads\s*[:\=]\s*(\d+)",
        ],
    ),
    (
        "num_layers",
        [
            r"(\d+)\s+(?:transformer\s+)?(?:encoder\s+)?(?:decoder\s+)?layers?",
            r"num_layers\s*[:\=]\s*(\d+)",
            r"depth\s*[:\=]\s*(\d+)",
        ],
    ),
    (
        "num_classes",
        [
            r"\b(\d+)\s*[-–]?\s*way\s+classification\b",
            r"\b(\d+)\s+classes\b",
            r"\bnum_classes\s*[:\=]\s*(\d+)\b",
        ],
    ),
]

# Connection / relationship extraction patterns
_CONNECTION_PATTERNS = [
    (r"\bskip[_\s]?connection\b", "skip"),
    (r"\bskip[_\s]?connect\b", "skip"),
    (r"\bresidual[_\s]?connection\b", "residual"),
    (r"\bshortcut[_\s]?connection\b", "skip"),
    (r"\bconcaten(?:ate|ation)\b", "concat"),
    (r"\bconcat\b", "concat"),
    (r"\bbranch\b", "branch"),
    (r"\bfork\b", "branch"),
    (r"\bmerge\b", "merge"),
    (r"\badd\b", "add"),
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
Text: "U-Net has an encoder with 3×3 convolutions producing 64 channels, followed by max pooling. Its decoder upsamples and concatenates encoder features via skip connections."
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

{operation_context}
{variant_rule}
### Now extract from this text:
Text: \"\"\"{text}\"\"\"

Return ONLY valid JSON — no explanation, no markdown fences.

Rules:
- "type" must be one of:
  conv2d, conv1d, convtranspose2d, depthwise_conv2d, linear,
  maxpool2d, avgpool2d, globalavgpool2d, upsample, flatten,
  batchnorm2d, layernorm, groupnorm, relu, leakyrelu, gelu, silu, dropout,
  multiheadattention, transformerblock, feedforward, patchembedding,
  positionalembedding, clstoken, sequence_pooling,
  residualblock, residual_add, concat
- "params": ONLY extract values EXPLICITLY stated in the text. Do NOT guess. Use these parameter names where the paper states them: channels, kernel_size, stride, padding, hidden_size, num_heads, num_layers, num_classes.
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


def _value_in_evidence(value: Any, evidence: str) -> bool:
    """True when a numeric replacement actually occurs in the evidence text."""
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return re.search(r"\b" + re.escape(str(value)) + r"\b", evidence) is not None
    return True  # non-numeric values are not checkable this way


def _repair_json(text: str) -> str:
    """Strip `//` and `/* */` comments and trailing commas, outside strings.

    Deliberately conservative: it tracks string state and escapes, so a `//`
    inside a quoted value (a URL, say) survives untouched.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < len(text) and text[index + 1] == "/":
            newline = text.find("\n", index)
            index = len(text) if newline == -1 else newline
            continue
        if char == "/" and index + 1 < len(text) and text[index + 1] == "*":
            close = text.find("*/", index + 2)
            index = len(text) if close == -1 else close + 2
            continue
        out.append(char)
        index += 1

    # Trailing commas: {"a": 1,} and [1, 2,]
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


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
        chunk_retriever: Callable[[str, list[str], int], list[str]] | None = None,
        variant: str | None = None,
        samples: int = 1,
    ):
        self.use_llm = use_llm and _HAS_LLM
        self.use_section_splitter = use_section_splitter
        self.use_retriever = use_retriever
        self.verify = verify and use_llm and _HAS_LLM
        self.max_context_chars = max_context_chars
        self.ontology = KnowledgeGraph()
        # Optional (query, texts, top_k) -> ranked texts callback. Lets a
        # caller plug in dense/hybrid retrieval over real page/section-aware
        # chunks without this module depending on any backend/network code.
        # Falls back to the pure BM25 path below when not supplied.
        self.chunk_retriever = chunk_retriever
        # Which model configuration to report when a paper describes several.
        # ViT states Base/Large/Huge in one table and runs ablations at other
        # sizes; without this the retriever had no reason to prefer the Base
        # row, and an ablation's "8 layers, D = 1024" was what reached the
        # model. This names the wanted variant; it never supplies its values.
        self.variant = (variant or "").strip() or None
        # Independent extraction samples to draw; the medoid is kept. Groq's
        # gpt-oss-120b is not deterministic at temperature=0 even with a seed
        # and low reasoning effort (same prompt: 3 vs 14 layers). Consensus
        # filters those outliers without merging specs from different draws.
        self.samples = max(1, int(samples))
        self.consensus: dict[str, Any] | None = None
        self.provider_models: list[str] = []
        self.provider_fallback = False
        # Verification provenance: what it said, and what it tried to overwrite.
        self.verification_response: str | None = None
        self.verification_reverted: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_text(
        self, text: str, source_chunks: list[dict[str, Any]] | None = None
    ) -> ConfigDict:
        """
        Full pipeline: raw text -> focused context -> extract -> verify -> normalize.

        `source_chunks` (optional) are the real page/section-aware chunks
        already computed upstream (core.utils.chunk_pages_with_provenance).
        When given, they're used for retrieval instead of re-chunking `text`
        from scratch with fixed-size windows.
        """
        focused = self._focus_text(text, source_chunks=source_chunks)

        extraction_method = "rule_based"
        extraction_reason = None
        try:
            if self.use_llm:
                raw = self._extract_with_llm_consensus(focused)
                extraction_method = "llm"
                if self.verify:
                    raw = self._verify_extraction(focused, raw)
                    extraction_method = "llm_verified"
            else:
                raw = self._extract_rule_based(focused)
        except Exception as exc:
            extraction_reason = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "LLM extraction failed; using rule-based fallback (%s)", extraction_reason
            )
            raw = self._extract_rule_based(focused)
            extraction_method = "rule_based_fallback"

        normalized = normalize_config(raw)
        normalized["extraction_method"] = extraction_method
        if self.provider_models:
            normalized["provider_models"] = self.provider_models
            normalized["provider_fallback"] = self.provider_fallback
        if extraction_reason is not None:
            normalized["extraction_reason"] = extraction_reason
        return normalized

    def extract_from_full_pdf(self, pdf_text: str) -> ConfigDict:
        """
        Entry point for full PDF text. Applies section splitting first.
        """
        focused = get_architecture_text(pdf_text, max_chars=self.max_context_chars)
        return self.extract_from_text(focused)

    # ------------------------------------------------------------------
    # Step 1: Text focusing
    # ------------------------------------------------------------------

    def _variant_rule(self) -> str:
        """Instruct the model which configuration to report, when it matters."""
        if not self.variant:
            return ""
        return (
            "\n### Model variant\n"
            "This paper describes several model configurations. Report ONLY the "
            f"values for {self.variant}. Do not merge values across variants, and "
            "do not take values from ablation studies or scaling experiments.\n"
        )

    def _retrieval_query(self) -> str:
        """The architecture query, plus the requested variant when there is one.

        A paper's variants table only ranks if the query mentions the variant;
        the same class of gap as the missing numeric tokens found in Phase 6.
        """
        if not self.variant:
            return _ARCHITECTURE_QUERY
        return f"{_ARCHITECTURE_QUERY} {self.variant}"

    def _rank_chunks(self, texts: list[str], top_k: int) -> list[str]:
        """Rank via the injected retriever, falling back to plain BM25."""
        if not texts or top_k <= 0:
            return []
        if self.chunk_retriever is not None:
            try:
                ranked = self.chunk_retriever(self._retrieval_query(), texts, top_k)
                if ranked:
                    return ranked
            except Exception:
                pass  # fall through to BM25, not to the whole legacy pipeline
        return retrieve_top_chunks(texts, top_k=top_k)

    def _select_focus_chunks(
        self,
        source_chunks: list[dict[str, Any]],
        total: int = _TOTAL_FOCUS_SLOTS,
        reserved: int = _RESERVED_STRUCTURED_SLOTS,
        expand_neighbors: bool = False,
        max_context_chars: int | None = None,
    ) -> list[str]:
        """Pick the chunks to focus on, reserving slots for tables and captions.

        Architecture hyperparameters live overwhelmingly in tables and figure
        captions, which score poorly against a prose architecture query and
        are exactly what MMR's diversity term evicts. Measured: EfficientNet
        parameterizes 18 of 18 convolutions *because* its architecture table
        was retrieved, while ViT's dimension table and U-Net's 64->128->256->512
        progression never reached the model at all.

        Ranking prose and structured chunks separately, with a bounded
        reservation, guarantees the tables get a hearing without letting them
        crowd out the prose that carries the layer order.
        """
        entries = [
            (index, chunk)
            for index, chunk in enumerate(source_chunks or [])
            if str(chunk.get("text") or "").strip()
        ]
        if not entries:
            return []

        structured = [
            (i, c)
            for i, c in entries
            if (
                str(c.get("chunk_type") or "") in _STRUCTURED_CHUNK_TYPES
                and _has_architectural_structured_content(str(c.get("text") or ""))
            )
        ]
        prose = [
            (i, c)
            for i, c in entries
            if str(c.get("chunk_type") or "") not in _STRUCTURED_CHUNK_TYPES
        ]

        def _take(pool: list[tuple[int, dict]], k: int) -> list[int]:
            if not pool or k <= 0:
                return []
            by_text: dict[str, list[int]] = {}
            for i, c in pool:
                by_text.setdefault(str(c.get("text") or ""), []).append(i)
            chosen: list[int] = []
            for ranked_text in self._rank_chunks([str(c.get("text") or "") for _, c in pool], k):
                bucket = by_text.get(ranked_text)
                if bucket:
                    chosen.append(bucket.pop(0))
            return chosen

        # A requested variant gets one guaranteed slot. Adding the variant to
        # the retrieval query is not enough on its own: broadening the query
        # with activation/normalisation vocabulary diluted "ViT-Base" enough
        # that the variants table dropped out again and the model went back to
        # reading an ablation's dimensions. Ranking cannot be trusted to surface
        # this row, for the same reason tables already get a reservation.
        variant_picks: list[int] = []
        if self.variant:
            needle = self.variant.lower()
            variant_pool = [
                (i, c) for i, c in entries if needle in str(c.get("text") or "").lower()
            ]
            variant_picks = _take(variant_pool, 1)

        claimed = set(variant_picks)
        structured = [(i, c) for i, c in structured if i not in claimed]
        prose = [(i, c) for i, c in prose if i not in claimed]

        remaining = max(0, total - len(variant_picks))
        structured_picks = _take(structured, min(reserved, remaining))
        prose_picks = _take(prose, remaining - len(structured_picks))

        # Reading order: the focused text is read by the model as a narrative,
        # matching retrieve_top_chunks' existing convention.
        lookup = dict(entries)
        ranked_order = variant_picks + structured_picks + prose_picks
        ranked_indices = set(ranked_order)
        if not expand_neighbors or max_context_chars is None:
            return [str(lookup[i].get("text") or "") for i in sorted(ranked_indices)]

        selected_indices = set(ranked_indices)

        # Expand along the paper's reading order, not list order. source_chunks
        # is prose, then every table, then every caption, so index+1 from a
        # table lands on an unrelated table from another page -- which is how
        # results tables were entering the context through pure list adjacency.
        # page/offset provenance is already on every chunk; use it.
        doc_sorted = sorted(
            entries,
            key=lambda item: (
                item[1].get("page") or 0,
                item[1].get("source_offset_start") or 0,
            ),
        )
        doc_pos = {index: pos for pos, (index, _) in enumerate(doc_sorted)}
        pos_index = {pos: index for pos, (index, _) in enumerate(doc_sorted)}

        def _merged_length(indices: set[int]) -> int:
            ordered = sorted(indices)
            return sum(len(str(lookup[i].get("text") or "")) for i in ordered) + (
                len(_FOCUS_SEPARATOR) * max(0, len(ordered) - 1)
            )

        def _add_if_within_budget(index: int) -> bool:
            if index not in lookup or index in selected_indices:
                return index in selected_indices
            candidate_indices = selected_indices | {index}
            if _merged_length(candidate_indices) > max_context_chars:
                return False
            selected_indices.add(index)
            return True

        for origin in ranked_order:
            origin_pos = doc_pos.get(origin)
            if origin_pos is None:
                continue
            for depth in range(1, _FORWARD_EXPANSION_DEPTH + 1):
                neighbour = pos_index.get(origin_pos + depth)
                if neighbour is None or not _add_if_within_budget(neighbour):
                    break
            previous = pos_index.get(origin_pos - 1)
            if previous is not None:
                _add_if_within_budget(previous)

        return [str(lookup[i].get("text") or "") for i in sorted(selected_indices)]

    def _focus_text(self, text: str, source_chunks: list[dict[str, Any]] | None = None) -> str:
        """Apply section splitting and BM25/hybrid retrieval to narrow the context."""
        if len(text) <= self.max_context_chars:
            return text

        if self.use_retriever:
            selected = self._select_focus_chunks(
                source_chunks or [],
                expand_neighbors=True,
                max_context_chars=self.max_context_chars,
            )
            if selected:
                merged = _FOCUS_SEPARATOR.join(selected)[: self.max_context_chars]
                if merged.strip():
                    return merged

        if self.use_section_splitter:
            text = get_architecture_text(text, max_chars=self.max_context_chars)

        if len(text) > self.max_context_chars and self.use_retriever:
            chunks = chunk_for_retrieval(text, chunk_size=1_200, overlap=200)
            text = retrieve_and_merge(chunks, top_k=6, max_chars=self.max_context_chars)

        return text

    # ------------------------------------------------------------------
    # Step 2: LLM extraction with few-shot prompt (R1)
    # ------------------------------------------------------------------

    def _extract_with_llm_consensus(self, text: str) -> dict[str, Any]:
        """Draw ``self.samples`` extractions; keep the one most like the others.

        Similarity is Jaccard over layer-type sets. The medoid is a real
        coherent spec from one draw, never a merge across draws. Ties go to
        the earliest sample so the choice is itself deterministic.
        """
        if self.samples <= 1:
            return self._extract_with_llm(text)
        # One malformed draw must not sink the paper. DenseNet's third draw
        # unrolled 540 layers and broke the JSON mid-string; the other two
        # parsed fine and the paper still fell to rule-based extraction.
        specs: list[dict[str, Any]] = []
        failures: list[str] = []
        for _ in range(self.samples):
            try:
                specs.append(self._extract_with_llm(text))
            except Exception as exc:  # noqa: BLE001 -- recorded, not hidden
                failures.append(f"{type(exc).__name__}: {exc}")
        if not specs:
            raise ValueError(f"all {self.samples} extraction draws failed: {failures[-1]}")
        if len(specs) == 1:
            self.consensus = {
                "samples": self.samples,
                "parsed": 1,
                "failed_draws": failures,
                "layer_counts": [len(specs[0].get("layers") or [])],
                "agreement": [1.0],
                "chosen": 0,
            }
            return specs[0]

        def types(spec: dict[str, Any]) -> set[str]:
            return {
                str(layer.get("type") or "").lower()
                for layer in (spec.get("layers") or [])
                if isinstance(layer, dict)
            }

        sets = [types(spec) for spec in specs]

        def jaccard(a: set[str], b: set[str]) -> float:
            union = a | b
            return len(a & b) / len(union) if union else 1.0

        agreement = [
            sum(jaccard(sets[i], sets[j]) for j in range(len(specs)) if j != i) / (len(specs) - 1)
            for i in range(len(specs))
        ]
        chosen = max(range(len(specs)), key=lambda i: (agreement[i], -i))
        self.consensus = {
            "samples": self.samples,
            "parsed": len(specs),
            "failed_draws": failures,
            "layer_counts": [len(spec.get("layers") or []) for spec in specs],
            "agreement": [round(a, 3) for a in agreement],
            "chosen": chosen,
        }
        return specs[chosen]

    def _extract_with_llm(self, text: str) -> dict[str, Any]:
        """Call LLM with few-shot prompt, KAG instructions, and connection instructions."""
        # KAG Entity Linking & Rule Extraction
        terms = self.ontology.identify_terms(text)
        graph_rules = self.ontology.get_context_for_terms(terms)

        prompt = _LLM_EXTRACTION_PROMPT.format(
            few_shot=_FEW_SHOT_EXAMPLES,
            graph_rules=graph_rules,
            operation_context=_operation_context(text),
            variant_rule=self._variant_rule(),
            text=text,
        )
        response = llm_complete(prompt, reasoning_effort=_EXTRACTION_REASONING_EFFORT)
        self._record_provider()
        return self._parse_json_response(response)

    # ------------------------------------------------------------------
    # Step 3: Self-correction loop (R3)
    # ------------------------------------------------------------------

    def _verify_extraction(self, original_text: str, extracted: dict[str, Any]) -> dict[str, Any]:
        """
        Ask the LLM to review its own extraction against the source text.
        Returns corrected dict, or original if correction fails.

        The prompt used to pass ``original_text[:4_000]``. ``original_text`` is
        already the focused context, bounded by ``max_context_chars``, so that
        second truncation only hid evidence: EfficientNet's supporting table row
        ("Conv1x1 & Pooling & FC ... 1280") sits at char 8379 of a 9906-char
        focus, while an unrelated "512" sits at char 1277. Verification saw the
        512 and not the row, and replaced a correct value with a wrong one.
        """
        try:
            prompt = _VERIFICATION_PROMPT.format(
                text=original_text,
                extracted=json.dumps(extracted, indent=2),
            )
            response = llm_complete(prompt, reasoning_effort=_EXTRACTION_REASONING_EFFORT)
            self._record_provider()
            self.verification_response = response
            corrected = self._parse_json_response(response)
            # Only accept correction if it has more or equal layers (no regression)
            corrected_layers = (corrected or {}).get("layers") or []
            extracted_layers = (extracted or {}).get("layers") or []
            if isinstance(corrected_layers, list) and isinstance(extracted_layers, list):
                if len(corrected_layers) >= len(extracted_layers):
                    return self._revert_unsupported_values(extracted, corrected, original_text)
        except Exception:
            pass
        return extracted

    def _revert_unsupported_values(
        self, extracted: dict[str, Any], corrected: dict[str, Any], evidence: str
    ) -> dict[str, Any]:
        """Keep verification's additions; refuse value swaps the evidence does not contain.

        The layer-count rule ("accept if layers did not decrease") cannot see a
        parameter being rewritten, so a correct dimension could be silently
        replaced by an invented one. A numeric replacement must appear in the
        evidence text to be accepted; otherwise the original value stands.
        """
        old_layers = (extracted or {}).get("layers") or []
        new_layers = (corrected or {}).get("layers") or []
        for index, new_layer in enumerate(new_layers):
            if index >= len(old_layers):
                break  # genuinely added layer; nothing to protect
            if not isinstance(new_layer, dict) or not isinstance(old_layers[index], dict):
                continue
            old_params = old_layers[index].get("params") or {}
            new_params = new_layer.get("params") or {}
            if not isinstance(old_params, dict) or not isinstance(new_params, dict):
                continue
            for key, old_value in old_params.items():
                if key not in new_params or new_params[key] == old_value:
                    continue
                if not _value_in_evidence(new_params[key], evidence):
                    rejected = new_params[key]
                    new_params[key] = old_value
                    self.verification_reverted.append(
                        {"layer": index, "param": key, "rejected": rejected, "kept": old_value}
                    )
        return corrected

    def _record_provider(self) -> None:
        """Retain completion provenance without changing llm_complete's return type."""
        provider = get_last_completion_model()
        if provider is not None:
            self.provider_models.append(provider)
            # litellm strips the routing prefix from resp.model: a call to
            # "groq/openai/gpt-oss-120b" reports back "openai/gpt-oss-120b".
            # A bare != comparison therefore flagged every single call as a
            # cross-provider fallback, which would have made the alarm
            # meaningless within one run. Compare the bare model id instead.
            if _bare_model_id(provider) != _bare_model_id(PRIMARY_MODEL):
                self.provider_fallback = True

    # ------------------------------------------------------------------
    # Step 4: Rule-based fallback (enhanced R4)
    # ------------------------------------------------------------------

    def _extract_rule_based(self, text: str) -> dict[str, Any]:
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
    def _parse_json_response(response: str) -> dict[str, Any]:
        """Parse JSON from an LLM response, tolerating common model quirks.

        Beyond raw JSON and markdown fences, this repairs two things models
        emit constantly and strict JSON rejects: `//` line comments (observed
        in production as `"params": {...},   // bottleneck 1x1`) and trailing
        commas. Both previously raised, and the caller's broad `except` turned
        that into a silent rule-based fallback -- a wrong spec reported as a
        result. Repair is attempted last, so well-formed output is untouched.
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        candidates: list[str] = []
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            candidates.append(match.group(1))
        # An unterminated fence (truncated output) still leaves usable JSON.
        opening = re.search(r"```(?:json)?\s*", response)
        if opening:
            candidates.append(response[opening.end() :])
        start, end = response.find("{"), response.rfind("}")
        if start != -1 and end > start:
            candidates.append(response[start : end + 1])

        for candidate in candidates:
            for attempt in (candidate, _repair_json(candidate)):
                try:
                    return json.loads(attempt)
                except json.JSONDecodeError:
                    continue

        raise ValueError("LLM did not return valid JSON")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _operation_context(text: str, limit: int = 8) -> str:
    """Return deterministic grounding for operations explicitly in ``text``."""
    try:
        canonical_names = find_mentioned(text)[:limit]
        if not canonical_names:
            return ""
        lines = ["KNOWN OPERATION DEFINITIONS (use these exact definitions; do not redefine them):"]
        for canonical_name in canonical_names:
            operation = OPERATIONS[canonical_name]
            syntax = operation["syntax"] or operation["functional"] or "no module form"
            lines.append(f"- {canonical_name}: {operation['formula']}  |  PyTorch: {syntax}")
        return "\n".join(lines)
    except Exception:
        return ""


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
            _stop = {
                "the",
                "a",
                "an",
                "this",
                "that",
                "with",
                "from",
                "starts",
                "uses",
                "has",
                "contains",
                "consists",
            }
            if name.lower() not in _stop and len(name) > 1:
                return name
    return "UnknownModel"


def _extract_layers(text: str) -> list[dict[str, Any]]:
    """Detect layers in order of appearance (position-based)."""
    multi_hits = _detect_multi_blocks(text)
    single_hits: list[tuple[int, str]] = []

    for pattern, canonical_type in _LAYER_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            multi_hits_2d = [(pos, typ) for pos, typ, _ in multi_hits]
            if not _pos_claimed(m.start(), single_hits + multi_hits_2d):
                single_hits.append((m.start(), canonical_type))

    all_hits: list[tuple[int, str, int]] = []
    for pos, typ, count in multi_hits:
        all_hits.append((pos, typ, count))
    for pos, typ in single_hits:
        all_hits.append((pos, typ, 1))

    all_hits.sort(key=lambda h: (h[0], h[1]))

    repeat_groups: dict[int, list[int]] = {}
    for idx, (pos, _, _) in enumerate(all_hits):
        repeat_groups.setdefault(pos, []).append(idx)

    layers = []
    for idx, (pos, layer_type, original_count) in enumerate(all_hits):
        params = _extract_params_near(text, pos)
        group_indices = repeat_groups[pos]
        if len(group_indices) > 1:
            local_index = group_indices.index(idx)
            actual_count = len(group_indices)
            repeat_group = layer_type if layer_type.endswith("block") else f"{layer_type}_block"
            params["_repeat_group"] = repeat_group
            params["_repeat_index"] = local_index
            params["_repeat_total"] = actual_count
            if original_count > 10:
                params["_repeat_truncated"] = True
        layers.append({"type": layer_type, "params": params})

    if not layers:
        layers = [{"type": "conv2d", "params": _extract_params_near(text, 0)}]
    return layers


def _detect_multi_blocks(text: str) -> list[tuple[int, str, int]]:
    """Detect 'N residual blocks', '3 conv layers', etc."""
    hits: list[tuple[int, str, int]] = []
    matched_ranges: list[tuple[int, int]] = []
    pattern = (
        r"(\d+)\s+(?:(?:conv|residual|dense|linear|attention|transformer)\w*\s+)?(blocks?|layers?)"
    )

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


def _pos_claimed(pos: int, hits: list[tuple[int, str]], tol: int = 10) -> bool:
    return any(abs(pos - h[0]) < tol for h in hits)


def _extract_params_near(text: str, pos: int) -> dict[str, Any]:
    """Extract explicit parameters in a wider window around pos (R4)."""
    # R4: expanded window — 40 chars lookback, 120 chars lookahead
    window_start = max(0, pos - 40)
    window_end = min(len(text), pos + 120)
    window = text[window_start:window_end]

    params: dict[str, Any] = {}
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


def _extract_connections(text: str, layers: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Build connections: sequential chain + detected skip/residual edges."""
    # Assign IDs first
    for i, layer in enumerate(layers):
        layer["id"] = f"layer_{i}"

    connections: list[tuple[str, str]] = []
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


def _extract_connection_types(text: str) -> dict[str, str]:
    """
    Detect connection type keywords in text.
    Returns a dict of keyword -> type for later graph annotation.
    """
    found: dict[str, str] = {}
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
    "conv": "conv2d",
    "convolution": "conv2d",
    "conv2d": "conv2d",
    "fc": "linear",
    "linear": "linear",
    "dense": "linear",
    "pool": "maxpool2d",
    "maxpool": "maxpool2d",
    "bn": "batchnorm2d",
    "batch norm": "batchnorm2d",
    "attention": "multiheadattention",
    "mha": "multiheadattention",
    "residual": "residualblock",
    "res block": "residualblock",
    "dropout": "dropout",
    "relu": "relu",
    "upsample": "upsample",
}


def extract_table_layers(text: str) -> list[dict[str, Any]]:
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
        params: dict[str, Any] = {}
        if numbers:
            params["channels"] = int(numbers[0])
        if len(numbers) > 1:
            params["kernel_size"] = int(numbers[1])

        layers.append({"type": canonical, "params": params})

    return layers
