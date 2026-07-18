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
        "dither_secondary_pattern": "checkerboard",
        "dither_mask_cleanup": False,
        "spatial_perturbation": 0.0,
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
    mask = encoder_helpers.generate_spatial_fusion_mask(6, 3, "spatial-dither-random", dither_ratio=0.5, seed=11, grid_shape=(2, 3))
    expected = torch.stack([sources[int(mask[index])][index] for index in range(6)])

    output = encoder_helpers.fuse_visual_token_sources(sources, config, "cpu", source_grids=[(2, 3)] * 3)

    assert output.dtype == torch.float16
    assert torch.equal(output, expected)


def test_nearest_grid_remap_selects_exact_tokens():
    short = torch.tensor([[0.0], [2.0]], dtype=torch.float16)
    long = torch.tensor([[10.0], [11.0], [12.0], [13.0]], dtype=torch.float16)

    output = encoder_helpers.fuse_visual_token_sources([long, short], _config(ratio=0.0), "cpu", source_grids=[(2, 2), (1, 2)])

    assert output.shape == (4, 1)
    assert output.dtype == torch.float16
    assert output.flatten().tolist() == [0.0, 2.0, 0.0, 2.0]


def test_deepstack_reuses_main_spatial_mask():
    config = _config(seed=23)
    cache = {}
    main_sources = [torch.zeros(16, 1), torch.ones(16, 1)]
    main = encoder_helpers.fuse_visual_token_sources(main_sources, config, "cpu", cache, 16, [(4, 4)] * 2)
    deepstack = {
        "a": [torch.full((16, 1), 10.0), torch.full((16, 1), 100.0)],
        "b": [torch.full((16, 1), 20.0), torch.full((16, 1), 200.0)],
    }

    layers = encoder_helpers.fuse_deepstack_layers(deepstack, config, "cpu", cache, 16, [(4, 4)] * 2)

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
        visual_grids={"a": (2, 2), "b": (2, 2)},
    )

    assert len(cache) == 1
    assert torch.equal(conditioning[0, 1:5, 0], saved[0][:, 0])


@pytest.mark.parametrize("method", ["index-consensus", "similarity-consensus", "unknown"])
def test_unsupported_methods_raise(method):
    with pytest.raises(ValueError, match="Unsupported visual fusion method"):
        encoder_helpers.fuse_visual_token_sources([torch.zeros(4, 1), torch.ones(4, 1)], _config(method), "cpu", source_grids=[(2, 2)] * 2)


def test_config_seed_and_legacy_call_compatibility():
    schema_inputs = UC_VisualFusionConfig.define_schema().inputs
    inputs = {value.id: value for value in schema_inputs}
    legacy = UC_VisualFusionConfig.execute("spatial-dither-random", 2, 0.5, False, "legacy.safetensors").args[0]
    seeded = UC_VisualFusionConfig.execute("spatial-dither-random", 2, 0.5, seed=123).args[0]

    assert [value.id for value in schema_inputs][-3:] == ["dither_secondary_pattern", "dither_mask_cleanup", "spatial_perturbation"]
    assert inputs["seed"].control_after_generate is True
    assert "index-consensus" not in inputs["visual_fusion_method"].options
    assert "similarity-consensus" not in inputs["visual_fusion_method"].options
    assert legacy["seed"] == 0
    assert legacy["save_path"] == "legacy.safetensors"
    assert seeded["seed"] == 123
    assert legacy["spatial_perturbation"] == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cached_cuda_mask_matches_cpu_raw_fusion():
    config = _config(seed=47)
    cache = {}
    gpu = encoder_helpers.fuse_visual_token_sources(
        [torch.zeros(64, 1, device="cuda"), torch.ones(64, 1, device="cuda")],
        config,
        "cuda",
        cache, source_grids=[(8, 8)] * 2,
    )
    cpu = encoder_helpers.fuse_visual_token_sources([torch.zeros(64, 1), torch.ones(64, 1)], config, "cuda", cache, source_grids=[(8, 8)] * 2)

    assert torch.equal(gpu.cpu(), cpu)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cleanup_mask_runs_on_cuda_without_cudnn_integer_convolution():
    mask = encoder_helpers.generate_spatial_fusion_mask(
        64, 3, "spatial-dither-random", dither_ratio=0.5, device="cuda", seed=5,
        grid_shape=(8, 8), dither_mask_cleanup=True,
    )
    assert mask.device.type == "cuda"
    assert mask.shape == (64,)


def test_equal_area_landscape_and_portrait_keep_canonical_orientation():
    landscape = torch.arange(6.0).reshape(2, 3, 1).flatten(0, 1)
    portrait = torch.arange(100.0, 106.0).reshape(3, 2, 1).flatten(0, 1)
    output = encoder_helpers.fuse_visual_token_sources(
        [landscape, portrait], _config(ratio=0.0), "cpu", source_grids=[(2, 3), (3, 2)]
    )
    expected = torch.nn.functional.interpolate(portrait.reshape(3, 2, 1).permute(2, 0, 1)[None], size=(2, 3), mode="nearest")[0].permute(1, 2, 0).flatten(0, 1)
    assert output.shape == (6, 1)
    assert torch.equal(output, expected)


def test_block_secondary_cleanup_endpoints_and_cache_separation():
    block = encoder_helpers.generate_spatial_fusion_mask(16, 4, "spatial-dither-random", 2, 0.0, grid_shape=(4, 4), dither_secondary_pattern="block-interleave")
    assert block.reshape(4, 4).tolist() == [[1, 1, 2, 2], [1, 1, 2, 2], [2, 2, 3, 3], [2, 2, 3, 3]]
    for ratio, expected in [(0.0, False), (1.0, True)]:
        cleaned = encoder_helpers.generate_spatial_fusion_mask(9, 3, "spatial-dither-random", dither_ratio=ratio, grid_shape=(3, 3), dither_mask_cleanup=True)
        assert cleaned.eq(0).all().item() is expected
    cache = {}
    sources = [torch.zeros(16, 1), torch.ones(16, 1)]
    encoder_helpers.fuse_visual_token_sources(sources, _config(seed=1), "cpu", cache, source_grids=[(4, 4)] * 2)
    changed = {**_config(seed=1), "dither_mask_cleanup": True}
    encoder_helpers.fuse_visual_token_sources(sources, changed, "cpu", cache, source_grids=[(4, 4)] * 2)
    perturbed = {**_config(seed=1), "spatial_perturbation": 0.5}
    encoder_helpers.fuse_visual_token_sources(sources, perturbed, "cpu", cache, source_grids=[(4, 4)] * 2)
    assert len(cache) == 3


@pytest.mark.parametrize("method", ["spatial-checkerboard", "spatial-block-interleave", "spatial-dither-random"])
def test_spatial_perturbation_is_seeded_and_preserves_source_counts(method):
    kwargs = {"grid_shape": (8, 8), "seed": 19, "spatial_perturbation": 0.5}
    base = encoder_helpers.generate_spatial_fusion_mask(64, 3, method, grid_shape=(8, 8), seed=19)
    first = encoder_helpers.generate_spatial_fusion_mask(64, 3, method, **kwargs)
    repeated = encoder_helpers.generate_spatial_fusion_mask(64, 3, method, **kwargs)
    changed_seed = encoder_helpers.generate_spatial_fusion_mask(64, 3, method, **{**kwargs, "seed": 20})

    assert torch.equal(first, repeated)
    assert not torch.equal(first, base)
    assert not torch.equal(first, changed_seed)
    assert torch.equal(torch.bincount(first, minlength=3), torch.bincount(base, minlength=3))
    assert first.ne(base).sum().item() == 32


def test_spatial_perturbation_saturates_without_ratio_drift():
    mask = torch.tensor([0] * 9 + [1])
    changed = encoder_helpers._perturb_spatial_assignments(mask, 1.0, seed=7)

    assert changed.ne(mask).sum().item() == 2
    assert torch.equal(torch.bincount(changed), torch.bincount(mask))
    assert torch.equal(encoder_helpers._perturb_spatial_assignments(mask, 0.0, seed=7), mask)


def test_spatial_perturbation_selects_only_exact_source_tokens():
    config = {**_config(method="spatial-checkerboard", seed=29), "spatial_perturbation": 0.75}
    sources = [torch.full((16, 2), float(index), dtype=torch.float16) for index in range(3)]
    mask = encoder_helpers.generate_spatial_fusion_mask(
        16, 3, "spatial-checkerboard", seed=29, grid_shape=(4, 4), spatial_perturbation=0.75,
    )
    output = encoder_helpers.fuse_visual_token_sources(sources, config, "cpu", source_grids=[(4, 4)] * 3)

    assert output.dtype == torch.float16
    assert torch.equal(output[:, 0], mask.to(dtype=output.dtype))
    assert torch.equal(output[:, 0], output[:, 1])


def test_cleanup_swaps_only_complementary_pairs_and_preserves_each_source():
    mask = torch.ones((7, 7), dtype=torch.long)
    mask[0, 0] = 0
    mask[3:6, 3:6] = 0
    mask[4, 4] = 2
    before_counts = torch.bincount(mask.flatten(), minlength=3)

    cleaned = encoder_helpers._cleanup_primary_pairs(mask)

    assert cleaned[0, 0].item() == 2
    assert cleaned[4, 4].item() == 0
    assert torch.equal(torch.bincount(cleaned.flatten(), minlength=3), before_counts)
    assert cleaned.ne(mask).sum().item() == 2


def test_cleanup_leaves_unpaired_sparse_islands_unchanged():
    sparse = torch.ones((8, 8), dtype=torch.long)
    sparse[1, 1] = 0
    sparse[6, 6] = 0

    assert torch.equal(encoder_helpers._cleanup_primary_pairs(sparse), sparse)


def test_combined_perturbation_and_cleanup_is_deterministic_and_balanced():
    kwargs = {
        "grid_shape": (12, 12),
        "seed": 37,
        "dither_ratio": 0.25,
        "spatial_perturbation": 0.4,
    }
    before = encoder_helpers.generate_spatial_fusion_mask(144, 4, "spatial-dither-random", **kwargs)
    cleaned = encoder_helpers.generate_spatial_fusion_mask(
        144, 4, "spatial-dither-random", dither_mask_cleanup=True, **kwargs,
    )
    repeated = encoder_helpers.generate_spatial_fusion_mask(
        144, 4, "spatial-dither-random", dither_mask_cleanup=True, **kwargs,
    )

    assert torch.equal(cleaned, repeated)
    assert torch.equal(torch.bincount(cleaned, minlength=4), torch.bincount(before, minlength=4))


@pytest.mark.parametrize("amount", [-0.01, 1.01])
def test_spatial_perturbation_rejects_invalid_amount(amount):
    with pytest.raises(ValueError, match="Spatial perturbation"):
        encoder_helpers.generate_spatial_fusion_mask(
            4, 2, "spatial-checkerboard", grid_shape=(2, 2), spatial_perturbation=amount,
        )


def test_missing_and_malformed_grids_rejected():
    sources = [torch.zeros(4, 1)]
    with pytest.raises(ValueError, match="explicit grid"):
        encoder_helpers.fuse_visual_token_sources(sources, _config(), "cpu")
    with pytest.raises(ValueError, match="inconsistent"):
        encoder_helpers.fuse_visual_token_sources(sources, _config(), "cpu", source_grids=[(1, 3)])
