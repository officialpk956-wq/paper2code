"""transformer_base extracted 6 encoder AND 6 decoder layers with causal
self-attention and cross-attention. The builder generated self.encoder only,
forward(x), and Linear(512, 1000) -- an encoder-only classifier from a seq2seq
paper. It ran cleanly, so sandbox validation passed it.
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from core.paper_to_code_generator import PaperToCodeGenerator  # noqa: E402
from core.transformer_model_builder import TransformerModelBuilder  # noqa: E402

BASE = {
    "model_family": "transformer",
    "input": {"vocab_size": 500, "max_seq_len": 64, "seq_len": 16},
    "output": {"num_classes": 10},
    "stem": {"params": {"d_model": 32}},
    "block": {"params": {"d_model": 32, "num_heads": 4, "ffn_dim": 64, "dropout": 0.0}},
    "stages": [{"repeats": 2}],
}


def test_encoder_only_schema_is_unchanged():
    """No decoder in the spec means the original classifier, exactly."""
    model = TransformerModelBuilder(BASE).eval()
    assert model.decoder is None
    assert model.fc.out_features == 10
    with torch.no_grad():
        out = model(torch.randint(0, 500, (2, 16)))
    assert tuple(out.shape) == (2, 10)


def test_decoder_schema_builds_a_seq2seq_model():
    model = TransformerModelBuilder({**BASE, "decoder_stages": [{"repeats": 2}]}).eval()
    assert model.decoder is not None and len(model.decoder) == 2
    # vocabulary projection, not class logits
    assert model.output_proj.out_features == 500
    with torch.no_grad():
        out = model(torch.randint(0, 500, (2, 16)), torch.randint(0, 500, (2, 5)))
    assert tuple(out.shape) == (2, 5, 500)


def test_decoder_has_both_self_and_cross_attention():
    model = TransformerModelBuilder({**BASE, "decoder_stages": [{"repeats": 2}]}).eval()
    names = [n for n, m in model.named_modules() if isinstance(m, nn.MultiheadAttention)]
    assert sum("decoder" in n and "self_attn" in n for n in names) == 2
    assert sum("decoder" in n and "cross_attn" in n for n in names) == 2


def test_decoder_self_attention_is_causal():
    """Without the mask a decoder can read future tokens -- the one thing the
    architecture exists to prevent."""
    from core.blocks_transformer import TransformerDecoderBlock

    block = TransformerDecoderBlock(d_model=8, num_heads=2, ffn_dim=16, dropout=0.0).eval()
    memory = torch.randn(1, 4, 8)
    x = torch.randn(1, 4, 8)
    with torch.no_grad():
        base = block(x, memory)
        changed = x.clone()
        changed[:, -1, :] += 10.0        # perturb only the LAST position
        after = block(changed, memory)
    # earlier positions must be unaffected by a later token
    assert torch.allclose(base[:, 0, :], after[:, 0, :], atol=1e-5)
    assert not torch.allclose(base[:, -1, :], after[:, -1, :], atol=1e-5)


def test_decoder_stages_are_derived_from_a_spec_with_decoder_blocks():
    spec = {
        "layers": [
            {"type": "transformerblock", "params": {"num_layers": 6, "subtype": "encoder"}},
            {"type": "transformerblock", "params": {"num_layers": 6, "subtype": "decoder"}},
        ]
    }
    prepared = PaperToCodeGenerator._prepare_builder_schema("transformer", spec)
    assert prepared.get("decoder_stages") == [{"repeats": 6}]


def test_spec_without_decoder_blocks_derives_none():
    spec = {"layers": [{"type": "transformerblock", "params": {"num_layers": 12}}]}
    prepared = PaperToCodeGenerator._prepare_builder_schema("transformer", spec)
    assert not prepared.get("decoder_stages")
