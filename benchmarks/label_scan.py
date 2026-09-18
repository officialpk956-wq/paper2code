"""Neutral evidence scan for benchmark labels.

For every label, for every canonical layer type NOT already expected, print
the first sentence in the paper that mentions it. Model output is never
consulted. A human then reads each hit and decides -- most hits are the paper
discussing other work or explicitly rejecting the component ("we do not use
dropout", "eliminating fully connected layers").

Usage:
    python -m benchmarks.label_scan                 # all labels
    python -m benchmarks.label_scan transformer_base
    python -m benchmarks.label_scan --type dropout  # one type across papers

Never calls an LLM. PDFs are cached under benchmarks/.cache/pdfs by the same
code path the live benchmark uses, so the scan reads exactly what the
extractor reads.
"""

import argparse
import re
import sys

from benchmarks.harness import LABELS_DIR, fetch_paper_chunks, load_label
from core.rag.normalizer import CANONICAL_TYPES

# Phrases a paper uses when describing ITS OWN architecture. Deliberately
# conservative: a hit is a prompt to read, not a decision.
EVIDENCE: dict[str, list[str]] = {
    "conv2d": [r"\bconvolution(?:al)? layers?\b", r"\b\d+\s*[x×]\s*\d+\s+conv"],
    "depthwise_conv2d": [r"depth-?wise (?:separable )?convolution"],
    "convtranspose2d": [r"transposed convolution", r"fractional(?:ly)?[- ]strided convolution", r"deconvolution"],
    "linear": [r"fully[- ]connected layer", r"\bFC layer", r"linear (?:layer|projection|transformation)"],
    "maxpool2d": [r"max[- ]?pool(?:ing)?\b"],
    "avgpool2d": [r"\b\d+\s*[x×]\s*\d+ average pool", r"average pooling layer"],
    "globalavgpool2d": [r"global average pool"],
    "batchnorm2d": [r"batch ?normali[sz]ation", r"\bbatchnorm\b"],
    "layernorm": [r"layer ?normali[sz]ation", r"\blayernorm\b"],
    "groupnorm": [r"group ?normali[sz]ation", r"\bgroup ?norm\b"],
    "relu": [r"\bReLU\b", r"rectified linear"],
    "leakyrelu": [r"leaky ?relu"],
    "gelu": [r"\bGELU\b"],
    "silu": [r"\bSiLU\b", r"\bSwish\b"],
    "tanh": [r"\bTanh\b"],
    "sigmoid": [r"\bsigmoid\b"],
    "softmax": [r"\bsoftmax\b"],
    "dropout": [r"\bdropout\b"],
    "upsample": [r"\bupsampl", r"up-?convolution"],
    "residualblock": [r"residual block", r"shortcut connection", r"identity (?:shortcut|mapping)"],
    "multiheadattention": [r"multi-?head(?:ed)? (?:self-)?attention", r"self-attention"],
    "transformerblock": [r"transformer (?:block|layer|encoder)"],
    "feedforward": [r"feed-?forward", r"\bFFN\b", r"position-wise"],
    "patchembedding": [r"patch embedding", r"linear projection of (?:the )?flattened patches"],
    "positionalembedding": [r"position(?:al)? (?:embedding|encoding)"],
    "embedding": [r"learned embeddings", r"token embedding", r"word embedding", r"input embedding"],
    "clstoken": [r"\[class\] token", r"\bclass token\b", r"\[CLS\]"],
    "concat": [r"concatenat"],
    "flatten": [r"\bflatten"],
    "crop": [r"\bcrop"],
    "mbconv": [r"\bMBConv\b", r"mobile inverted bottleneck"],
}


def scan_label(label: dict, only_type: str | None = None) -> list[tuple[str, str]]:
    """(type, sentence) for each candidate type with evidence in the paper."""
    text, _pages, _chunks = fetch_paper_chunks(str(label["source"]))
    have = {t.lower() for t in label["expected"].get("layer_types") or []}
    hits: list[tuple[str, str]] = []
    for layer_type in sorted(CANONICAL_TYPES):
        if layer_type in have or (only_type and layer_type != only_type):
            continue
        for pattern in EVIDENCE.get(layer_type, []):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start = max(0, match.start() - 90)
                snippet = text[start:match.end() + 90].replace("\n", " ")
                hits.append((layer_type, snippet))
                break
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("labels", nargs="*", help="label ids (default: all)")
    parser.add_argument("--type", help="scan one canonical type only")
    args = parser.parse_args(argv)

    if args.type and args.type not in CANONICAL_TYPES:
        parser.error(f"{args.type!r} is not a canonical type")

    paths = [LABELS_DIR / f"{lid}.json" for lid in args.labels] or sorted(LABELS_DIR.glob("*.json"))
    missing = [p for p in paths if not p.exists()]
    if missing:
        parser.error("no such label: " + ", ".join(p.stem for p in missing))

    scanned = 0
    for path in paths:
        label = load_label(path)
        # A paper that cannot be read is an error, not a skip. The 2026-09-17
        # scan silently dropped transformer_base and nobody counted sections.
        hits = scan_label(label, only_type=args.type)
        scanned += 1
        have = sorted(t.lower() for t in label["expected"].get("layer_types") or [])
        print(f"\n=== {label['paper_id']}  (has: {have}) ===")
        if not hits:
            print("  (no candidates)")
        for layer_type, snippet in hits:
            print(f"  {layer_type:20} {snippet.encode('ascii', 'replace').decode()}")

    print(f"\nscanned {scanned} of {len(paths)} label(s)")
    return 0 if scanned == len(paths) else 1


if __name__ == "__main__":
    sys.exit(main())
