import pathlib
import sys
import types

import pytest
import torch


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_test import encoder_helpers
    from utils_collection_test.encoder_nodes import UC_VisualFusionConfig
finally:
    cli_args.cpu = prior_cpu


def _config(method="spatial-dither-random", ratio=0.5, seed=0):
    return {
        "visual_fusion_method": method,
        "visual_block_size": 2,
        "dither_ratio": ratio,
        "seed": seed,
    }


def test_dither_seed_and_secondary_checkerboard():
    first = encoder_helpers.generate_spatial_fusion_mask(256, 3, "spatial-dither-random", dither_ratio=0.5, seed=7)
    repeated = encoder_helpers.generate_spatial_fusion_mask(256, 3, "spatial-dither-random", dither_ratio=0.5, seed=7)
    changed = encoder_helpers.generate_spatial_fusion_mask(256, 3, "spatial-dither-random", dither_ratio=0.5, seed=8)

    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)
    assert encoder_helpers.generate_spatial_fusion_mask(6, 3, "spatial-dither-random", dither_ratio=0.0).tolist() == [1, 2, 2, 1, 1, 2]
    assert encoder_helpers.generate_spatial_fusion_mask(6, 3, "spatial-dither-random", dither_ratio=1.0).tolist() == [0] * 6


def test_old_mask_device_position_remains_compatible():
    mask = encoder_helpers.generate_spatial_fusion_mask(4, 2, "spatial-checkerboard", 2, 0.5, "cpu")
    assert mask.device.type == "cpu"
    assert mask.tolist() == [0, 1, 1, 0]


def test_spatial_fusion_matches_mask_and_preserves_dtype():
    sources = [
        torch.arange(12, dtype=torch.float16).reshape(6, 2),
        torch.arange(100, 112, dtype=torch.float16).reshape(6, 2),
        torch.arange(200, 212, dtype=torch.float16).reshape(6, 2),
    ]
    config = _config(seed=11)
    mask = encoder_helpers.generate_spatial_fusion_mask(6, 3, "spatial-dither-random", dither_ratio=0.5, seed=11)
    expected = torch.stack([sources[int(mask[index])][index] for index in range(6)])

    output = encoder_helpers.fuse_visual_token_sources(sources, config, "cpu")

    assert output.dtype == torch.float16
    assert torch.equal(output, expected)


def test_interpolation_only_aligns_unequal_lengths():
    short = torch.tensor([[0.0], [2.0]], dtype=torch.float16)
    long = torch.tensor([[10.0], [11.0], [12.0], [13.0]], dtype=torch.float16)

    output = encoder_helpers.fuse_visual_token_sources([short, long], _config(ratio=1.0), "cpu")

    assert output.shape == (4, 1)
    assert output.dtype == torch.float16
    assert output.flatten().tolist() == [0.0, 0.5, 1.5, 2.0]


def test_deepstack_reuses_main_spatial_mask():
    config = _config(seed=23)
    cache = {}
    main_sources = [torch.zeros(16, 1), torch.ones(16, 1)]
    main = encoder_helpers.fuse_visual_token_sources(main_sources, config, "cpu", cache, 16)
    deepstack = {
        "a": [torch.full((16, 1), 10.0), torch.full((16, 1), 100.0)],
        "b": [torch.full((16, 1), 20.0), torch.full((16, 1), 200.0)],
    }

    layers = encoder_helpers.fuse_deepstack_layers(deepstack, config, "cpu", cache, 16)

    assert len(cache) == 1
    assert torch.equal(main.bool(), layers[0].eq(20.0))
    assert torch.equal(main.bool(), layers[1].eq(200.0))


def test_saved_raw_embedding_uses_active_conditioning_mask(monkeypatch):
    config = {**_config(seed=31), "save_blended_embeds": True}
    cache = {}
    sequence_tensors = {
        "a": torch.zeros(1, 6, 1),
        "b": torch.ones(1, 6, 1),
    }
    visual_ranges = {"a": (1, 5), "b": (1, 5)}
    tokens = {
        "a": {"qwen3vl_4b": [[(0, 1.0)]]},
        "b": {"qwen3vl_4b": [[(1, 1.0)]]},
    }

    class ClipModel:
        def process_tokens(self, tokens_only, device):
            value = float(tokens_only[0][0])
            return torch.full((1, 6, 2), value, device=device), None, None, None

    class Clip:
        cond_stage_model = ClipModel()

    saved = []
    monkeypatch.setattr(encoder_helpers, "save_blended_visual_embeddings", lambda tensors, config, key: saved.append(tensors[0]))

    conditioning, _ = encoder_helpers.evaluate_conditioning_consensus_blend(
        sequence_tensors,
        {},
        config,
        "cpu",
        visual_ranges,
        clip=Clip(),
        tokens_dict=tokens,
        mask_cache=cache,
    )

    assert len(cache) == 1
    assert torch.equal(conditioning[0, 1:5, 0], saved[0][:, 0])


@pytest.mark.parametrize("method", ["index-consensus", "similarity-consensus", "unknown"])
def test_unsupported_methods_raise(method):
    with pytest.raises(ValueError, match="Unsupported visual fusion method"):
        encoder_helpers.fuse_visual_token_sources([torch.zeros(4, 1), torch.ones(4, 1)], _config(method), "cpu")


def test_config_seed_and_legacy_call_compatibility():
    schema_inputs = UC_VisualFusionConfig.define_schema().inputs
    inputs = {value.id: value for value in schema_inputs}
    legacy = UC_VisualFusionConfig.execute("spatial-dither-random", 2, 0.5, False, "legacy.safetensors").args[0]
    seeded = UC_VisualFusionConfig.execute("spatial-dither-random", 2, 0.5, seed=123).args[0]

    assert [value.id for value in schema_inputs] == ["visual_fusion_method", "visual_block_size", "dither_ratio", "save_blended_embeds", "save_path", "seed"]
    assert inputs["seed"].control_after_generate is True
    assert "index-consensus" not in inputs["visual_fusion_method"].options
    assert "similarity-consensus" not in inputs["visual_fusion_method"].options
    assert legacy["seed"] == 0
    assert legacy["save_path"] == "legacy.safetensors"
    assert seeded["seed"] == 123


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cached_cuda_mask_matches_cpu_raw_fusion():
    config = _config(seed=47)
    cache = {}
    gpu = encoder_helpers.fuse_visual_token_sources(
        [torch.zeros(64, 1, device="cuda"), torch.ones(64, 1, device="cuda")],
        config,
        "cuda",
        cache,
    )
    cpu = encoder_helpers.fuse_visual_token_sources([torch.zeros(64, 1), torch.ones(64, 1)], config, "cuda", cache)

    assert torch.equal(gpu.cpu(), cpu)
