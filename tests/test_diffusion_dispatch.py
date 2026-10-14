"""DDPM reached the U-Net builder and produced a segmentation net: no timestep
conditioning, no GroupNorm, forward(x) with nowhere to put the denoising step.
Two independent faults, both pinned here.
"""
import pytest

from core.architecture_graph import ArchitectureGraph, GraphNode
from core.classification import classify_architecture


def _graph(types):
    g = ArchitectureGraph(name="t", metadata={})
    for i, t in enumerate(types):
        g.add_node(GraphNode(id=f"n{i}", type=t, label=t, params={}))
    return g


def test_timestep_conditioned_unet_is_diffusion_not_unet():
    """The paper's words ("sinusoidal", "timestep") normalise to the canonical
    type positionalembedding, so the old keyword rule never matched."""
    g = _graph(["positionalembedding", "conv2d", "groupnorm", "residualblock",
                "multiheadattention", "maxpool2d", "upsample", "linear"])
    assert classify_architecture(g) == "diffusion"


def test_groupnorm_plus_attention_unet_is_diffusion():
    g = _graph(["conv2d", "groupnorm", "multiheadattention", "upsample"])
    assert classify_architecture(g) == "diffusion"


def test_plain_segmentation_unet_is_still_unet():
    """Regression guard: the broadened rule must not swallow real U-Nets."""
    g = _graph(["conv2d", "maxpool2d", "upsample", "concat", "relu", "crop"])
    assert classify_architecture(g) == "unet"


def test_diffusion_resolves_to_the_ddpm_builder():
    """classify_architecture returns "diffusion"; the builder map keyed the
    entry "ddpm", so the builder was unreachable by the name actually used."""
    from core.paper_to_code_generator import PaperToCodeGenerator

    code = PaperToCodeGenerator()._builder_code("diffusion", {})
    assert "class DDPMBuilder" in code
    # Its dependencies must be emitted too, or the module cannot be imported.
    assert "class SinusoidalPositionEmbeddings" in code
    assert "class ResnetBlock" in code


def test_generated_diffusion_module_is_importable_and_time_conditioned():
    torch = pytest.importorskip("torch")
    import importlib.util
    import inspect

    from core.paper_to_code_generator import PaperToCodeGenerator

    code = PaperToCodeGenerator()._builder_code("diffusion", {})
    import tempfile, os

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "gen_diffusion.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(code)
        spec = importlib.util.spec_from_file_location("gen_diffusion", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    model = module.DDPMBuilder(channels=3, dim=16).eval()
    params = dict(inspect.signature(model.forward).parameters)
    assert "time" in params, "a diffusion model must accept a timestep"
    kinds = {type(m).__name__ for _, m in model.named_modules()}
    assert "SinusoidalPositionEmbeddings" in kinds
    assert "GroupNorm" in kinds
    with torch.no_grad():
        out = model(torch.randn(1, 3, 32, 32), torch.randint(0, 1000, (1,)))
    assert out.shape[0] == 1
