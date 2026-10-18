"""Parameter and activation-memory estimates for the Labs graphs (ResNet, U-Net, ViT, Transformer).

The lab graphs are schematic (one node per stage), so a node on its own does not carry enough
to count weights: a ResNet stage knows its output width but not its input width, a decoder block
only names the encoder it skips from. This walks the graph in order, tracking channel width and
spatial size, and applies the standard formulas for each family.

Planning estimates: weights plus norm/bias terms, and fp32 activations (sum of layer outputs)
at the given batch size. Node types it does not know fall back to the generic estimator.
"""

import re

from core.architecture_graph import ArchitectureGraph, GraphNode
from core.metrics_estimator import _estimate_node_params, _extract_numeric_value

_BYTES = 4  # fp32
_POOLS = ("maxpool", "maxpool2d", "avgpool", "avgpool2d")


def _num(value, default: int = 0) -> int:
    return _extract_numeric_value(value, default)


def _in_out(p: dict, current: int) -> tuple[int, int]:
    """(in, out) channels from params like "3 → 64", 64, or filters=64."""
    raw = p.get("channels", p.get("out_channels", p.get("filters")))
    nums = re.findall(r"\d+", str(raw)) if raw is not None else []
    if len(nums) >= 2:
        return int(nums[0]), int(nums[-1])
    if nums:
        return current, int(nums[0])
    return current, current


def _kernel(node: GraphNode) -> int:
    p = node.params or {}
    if p.get("kernel_size"):
        return _num(p["kernel_size"], 3)
    label = (node.label or "").lower()
    if "output" in label:
        return 1  # 1x1 projection to the output classes
    m = re.search(r"(\d+)\s*[×x]\s*\d+", label)
    return int(m.group(1)) if m else 3


def estimate_params_and_memory(
    graph: ArchitectureGraph, batch_size: int = 1, input_spatial: int = 224
) -> tuple[int, float]:
    """Return (parameter count, activation memory in MB) for a lab graph."""
    params = 0
    acts = 0  # floats at batch size 1
    c = 3  # current channel / feature width
    h = input_spatial  # current spatial size (square)
    seq = 0  # token count once in token space
    res_stages = 0  # ResNet stages seen so far
    skips: dict[str, tuple[int, int]] = {}  # U-Net encoder id -> (filters, spatial)

    for node in graph.nodes:
        t = (node.type or "").lower()
        p = node.params or {}

        if t == "conv2d":
            cin, cout = _in_out(p, c)
            k = _kernel(node)
            params += cin * cout * k * k + cout
            h = max(1, h // max(1, _num(p.get("stride"), 1)))
            acts += cout * h * h
            c = cout

        elif t in _POOLS:
            h = max(1, h // max(1, _num(p.get("stride"), 2)))
            acts += c * h * h

        elif t == "residualblock":  # ResNet basic blocks: two 3x3 convs + two BN each
            blocks = max(1, _num(p.get("blocks"), 1))
            cout = _num(p.get("channels"), c)
            for b in range(blocks):
                cin = c if b == 0 else cout
                if b == 0 and cin != cout:
                    if res_stages > 0:
                        h = max(1, h // 2)  # later widening stages downsample; stage 1 does not
                    params += cin * cout  # 1x1 projection shortcut
                params += 9 * cin * cout + 9 * cout * cout + 4 * cout
                acts += 2 * cout * h * h
            c = cout
            res_stages += 1

        elif t in ("encoderblock", "bottleneck"):  # U-Net: two 3x3 convs (+ 2x2 pool)
            f = _num(p.get("filters"), c)
            params += 9 * c * f + 9 * f * f + 2 * f
            acts += 2 * f * h * h
            c = f
            if t == "encoderblock":
                skips[node.id] = (f, h)
                h = max(1, h // 2)

        elif t == "decoderblock":  # U-Net: 2x2 up-conv, concat with skip, two 3x3 convs
            sf, sh = skips.get(p.get("skip", ""), (max(1, c // 2), h * 2))
            params += 4 * c * sf + sf + 27 * sf * sf + 2 * sf
            h = sh
            acts += 5 * sf * h * h
            c = sf

        elif t in ("patchembedding", "patch_embedding") or (
            t == "embedding" and "vocab_size" not in p
        ):
            d = _num(p.get("embed_dim", p.get("hidden_size", p.get("d_model"))), c)
            patch = _num(p.get("patch_size"), 0) or _num(p.get("patch"), 16)
            n = (input_spatial // max(1, patch)) ** 2
            seq = n + 1  # + class token
            params += patch * patch * 3 * d + d + d + seq * d  # proj, cls, positional
            acts += seq * d
            c = d

        elif t == "embedding":  # token embedding table + positional table
            d = _num(p.get("embed_dim", p.get("d_model")), c)
            seq = _num(p.get("seq_len"), 128)
            params += _num(p.get("vocab_size")) * d + seq * d
            acts += seq * d
            c = d

        elif t == "transformerencoder":  # attention 4d^2 + MLP 8d^2 + biases + 2 LN
            d = _num(p.get("d_model"), c)
            heads = _num(p.get("heads"), 8)
            n = seq or 197
            params += 12 * d * d + 13 * d
            acts += 10 * n * d + heads * n * n
            c = d

        elif t in ("multiheadattention", "mhsa"):  # e.g. added by the Add Attention mutation
            d = _num(p.get("d_model"), c)
            heads = _num(p.get("heads", p.get("num_heads")), 8)
            n = seq or h * h
            params += 4 * d * d + 4 * d
            acts += 4 * n * d + heads * n * n

        elif t in ("mlphead", "linear"):
            out = _num(p.get("classes", p.get("out_features")), 1000)
            params += c * out + out
            acts += out
            c = out

        else:
            params += _estimate_node_params(node)

    return params, round(acts * batch_size * _BYTES / 2**20, 2)
