import pathlib
import sys
import types

import pytest
import torch


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_high_resolution_tiling_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_high_resolution_tiling_test.image_nodes import (
    HighResolutionTileLayout,
    UC_HighResolutionTileAccumulator,
    UC_HighResolutionTileSplit,
)
from utils_collection_high_resolution_tiling_test.tile_helpers import (
    accumulate_tile_images,
    apply_tile_differential_diffusion,
    build_tile_records,
    split_and_encode_tiles,
    tile_weight_mask,
)


class MockVAE:
    def __init__(self, compression=8):
        self.compression = compression
        self.calls = []

    def spacial_compression_encode(self):
        return self.compression

    def encode(self, image):
        self.calls.append(tuple(image.shape))
        return torch.zeros(
            (
                image.shape[0],
                4,
                image.shape[1] // self.compression,
                image.shape[2] // self.compression,
            ),
            dtype=image.dtype,
            device=image.device,
        )


class MockFiveDimensionalImageVAE(MockVAE):
    def encode(self, image):
        self.calls.append(tuple(image.shape))
        return torch.zeros(
            (
                image.shape[0],
                16,
                1,
                image.shape[1] // self.compression,
                image.shape[2] // self.compression,
            ),
            dtype=image.dtype,
            device=image.device,
        )


class MockModel:
    def __init__(self):
        self.denoise_mask_function = None

    def clone(self):
        return MockModel()

    def set_model_denoise_mask_function(self, function):
        self.denoise_mask_function = function


class MockSampling:
    sigma_min = 0.0

    @staticmethod
    def timestep(sigma):
        return sigma


class MockSamplingModel:
    def __init__(self):
        self.inner_model = types.SimpleNamespace(model_sampling=MockSampling())


def _split(image, vae=None, **overrides):
    settings = {
        "tile_mode": "tile_size",
        "tile_width": 8,
        "tile_height": 8,
        "rows": 2,
        "columns": 2,
        "overlap": 2,
        "mask_profile": "cosine",
        "feather_width": 1.0,
        "mask_strength": 1.0,
    }
    settings.update(overrides)
    return split_and_encode_tiles(image, vae or MockVAE(), **settings)


def test_public_schema_and_list_contract():
    split_schema = UC_HighResolutionTileSplit.define_schema()
    accumulator_schema = UC_HighResolutionTileAccumulator.define_schema()

    assert split_schema.node_id == "UC_HighResolutionTileSplit"
    assert split_schema.display_name == "High Resolution Tile Split & VAE Encode"
    assert [output.id for output in split_schema.outputs] == [
        "tile_images",
        "tile_latents",
        "tile_layout",
        "model",
    ]
    assert [output.is_output_list for output in split_schema.outputs] == [
        True,
        True,
        False,
        False,
    ]
    assert split_schema.outputs[2].io_type == "UC_HIGH_RES_TILE_LAYOUT"
    assert HighResolutionTileLayout.io_type == "UC_HIGH_RES_TILE_LAYOUT"
    input_ids = [value.id for value in split_schema.inputs]
    assert input_ids[-2:] == [
        "differential_diffusion_mode",
        "differential_diffusion_value",
    ]

    assert accumulator_schema.node_id == "UC_HighResolutionTileAccumulator"
    assert accumulator_schema.display_name == "High Resolution Tile Accumulator"
    assert accumulator_schema.is_input_list is True
    assert [value.id for value in accumulator_schema.inputs] == [
        "images",
        "tile_layout",
    ]


def test_fixed_size_coordinates_preserve_exact_overlap_and_row_major_order():
    records, rows, columns = build_tile_records(
        height=10,
        width=14,
        tile_mode="tile_size",
        tile_width=8,
        tile_height=8,
        rows=99,
        columns=99,
        overlap=2,
    )

    assert (rows, columns) == (2, 2)
    assert [
        (record["x0"], record["y0"], record["x1"], record["y1"])
        for record in records
    ] == [
        (0, 0, 8, 8),
        (6, 0, 14, 8),
        (0, 6, 8, 10),
        (6, 6, 14, 10),
    ]
    assert records[0]["right_overlap"] == 2
    assert records[1]["left_overlap"] == 2
    assert records[0]["bottom_overlap"] == 2
    assert records[2]["top_overlap"] == 2


def test_grid_coordinates_use_requested_shared_overlap():
    records, rows, columns = build_tile_records(
        height=13,
        width=17,
        tile_mode="grid",
        tile_width=1,
        tile_height=1,
        rows=2,
        columns=3,
        overlap=2,
    )

    assert (rows, columns) == (2, 3)
    assert len(records) == 6
    for record in records:
        if record["column"] > 0:
            assert record["left_overlap"] == 2
        if record["column"] < columns - 1:
            assert record["right_overlap"] == 2
        if record["row"] > 0:
            assert record["top_overlap"] == 2
        if record["row"] < rows - 1:
            assert record["bottom_overlap"] == 2


@pytest.mark.parametrize("profile", ["linear", "cosine"])
def test_masks_protect_only_internal_edges(profile):
    record = {
        "height": 6,
        "width": 8,
        "left_overlap": 0,
        "right_overlap": 4,
        "top_overlap": 0,
        "bottom_overlap": 0,
    }
    mask = tile_weight_mask(
        record,
        profile,
        feather_width=1.0,
        strength=1.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert mask.shape == (6, 8)
    assert torch.all(mask[:, :4] == 1)
    assert torch.all(mask[:, -1] == 0)
    assert torch.all(mask[:, -4] == 1)


def test_mask_strength_controls_protected_edge_floor():
    record = {
        "height": 4,
        "width": 6,
        "left_overlap": 3,
        "right_overlap": 0,
        "top_overlap": 0,
        "bottom_overlap": 0,
    }
    mask = tile_weight_mask(
        record,
        "linear",
        feather_width=1.0,
        strength=0.25,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.allclose(mask[:, 0], torch.full((4,), 0.75))
    assert torch.all(mask[:, -1] == 1)


def test_feather_width_changes_transition_geometry():
    record = {
        "height": 2,
        "width": 8,
        "left_overlap": 0,
        "right_overlap": 4,
        "top_overlap": 0,
        "bottom_overlap": 0,
    }
    full = tile_weight_mask(
        record, "linear", 1.0, 1.0, torch.device("cpu"), torch.float32
    )
    half = tile_weight_mask(
        record, "linear", 0.5, 1.0, torch.device("cpu"), torch.float32
    )

    assert full[0, -3] < 1
    assert half[0, -3] == 1
    assert full[0, -1] == half[0, -1] == 0


def test_split_encodes_each_tile_sequentially_and_attaches_masks():
    image = torch.rand(1, 10, 14, 3)
    vae = MockVAE(compression=8)
    images, latents, layout = _split(image, vae)

    assert len(images) == len(latents) == len(layout["tiles"]) == 4
    assert all(call[0] == 1 for call in vae.calls)
    assert vae.calls == [
        (1, 8, 8, 3),
        (1, 8, 8, 3),
        (1, 8, 8, 3),
        (1, 8, 8, 3),
    ]
    assert images[2].shape == (1, 4, 8, 3)
    assert latents[2]["samples"].shape == (1, 4, 1, 1)
    assert latents[2]["noise_mask"].shape == (1, 1, 8, 8)
    assert torch.all(latents[2]["noise_mask"][:, :, 4:, :] == 0)


def test_split_accepts_image_vae_with_singleton_temporal_latent_axis():
    image = torch.rand(1, 8, 14, 3)
    images, latents, layout = _split(
        image,
        MockFiveDimensionalImageVAE(compression=8),
    )

    assert len(images) == len(latents) == len(layout["tiles"]) == 2
    assert all(latent["samples"].shape == (1, 16, 1, 1, 1) for latent in latents)
    assert all(latent["noise_mask"].ndim == 4 for latent in latents)


@pytest.mark.parametrize(
    ("shape", "settings"),
    [
        (
            (1, 19, 23, 3),
            {
                "tile_mode": "tile_size",
                "tile_width": 12,
                "tile_height": 11,
                "overlap": 4,
            },
        ),
        (
            (1, 13, 17, 3),
            {
                "tile_mode": "grid",
                "rows": 2,
                "columns": 3,
                "overlap": 2,
            },
        ),
        (
            (1, 7, 11, 3),
            {
                "tile_mode": "tile_size",
                "tile_width": 32,
                "tile_height": 32,
                "overlap": 4,
            },
        ),
    ],
)
def test_identity_tiles_reconstruct_exact_source(shape, settings):
    image = torch.rand(shape, dtype=torch.float64)
    images, _, layout = _split(image, **settings)

    reconstructed = accumulate_tile_images(images, [layout])

    assert reconstructed.shape == image.shape
    assert reconstructed.dtype == image.dtype
    assert torch.allclose(reconstructed, image, atol=1e-6)


def test_accumulator_normalizes_different_overlap_values():
    image = torch.zeros(1, 8, 14, 3)
    images, _, layout = _split(
        image,
        tile_width=8,
        tile_height=8,
        overlap=2,
        mask_profile="linear",
    )
    images[0] = torch.ones_like(images[0])
    images[1] = torch.zeros_like(images[1])

    reconstructed = accumulate_tile_images(images, layout)

    assert torch.all(reconstructed[:, :, :6] == 1)
    assert torch.all(reconstructed[:, :, 8:] == 0)
    assert torch.allclose(
        reconstructed[0, 0, 6:8, 0],
        torch.tensor([1.0, 0.0]),
        atol=1e-6,
    )


def test_single_image_and_validation_errors_are_actionable():
    with pytest.raises(ValueError, match="exactly one image"):
        _split(torch.zeros(2, 16, 16, 3))

    with pytest.raises(ValueError, match="smaller than tile width"):
        _split(
            torch.zeros(1, 16, 16, 3),
            tile_width=8,
            tile_height=8,
            overlap=8,
        )

    images, _, layout = _split(torch.zeros(1, 16, 16, 3))
    with pytest.raises(ValueError, match="expected .* images"):
        accumulate_tile_images(images[:-1], layout)


def test_node_execute_returns_synchronized_outputs():
    image = torch.rand(1, 8, 14, 3)
    output = UC_HighResolutionTileSplit.execute(
        image,
        MockVAE(),
        "tile_size",
        8,
        8,
        2,
        2,
        2,
        "cosine",
        1.0,
        1.0,
    ).args

    assert len(output[0]) == len(output[1]) == len(output[2]["tiles"]) == 2
    merged = UC_HighResolutionTileAccumulator.execute(
        output[0],
        [output[2]],
    ).args[0]
    assert torch.allclose(merged, image, atol=1e-6)
    assert output[3] is None


def test_tile_differential_diffusion_off_preserves_model_identity():
    model = MockModel()

    assert apply_tile_differential_diffusion(model, "off", 1.0) is model


def test_tile_differential_diffusion_core_uses_core_mask_behavior():
    model = apply_tile_differential_diffusion(MockModel(), "core", 0.5)
    mask = torch.tensor([[[[0.2, 0.6]]]])
    result = model.denoise_mask_function(
        torch.tensor([0.5]),
        mask,
        {
            "model": MockSamplingModel(),
            "sigmas": torch.tensor([1.0, 0.0]),
        },
    )

    assert torch.allclose(result, torch.tensor([[[[0.1, 0.8]]]]))


def test_tile_differential_diffusion_advanced_uses_threshold_multiplier():
    model = apply_tile_differential_diffusion(
        MockModel(),
        "advanced",
        2.0,
    )
    mask = torch.tensor([[[[0.2, 0.3]]]])
    result = model.denoise_mask_function(
        torch.tensor([0.5]),
        mask,
        {
            "model": MockSamplingModel(),
            "sigmas": torch.tensor([1.0, 0.0]),
        },
    )

    assert torch.equal(result, torch.tensor([[[[0.0, 1.0]]]]))


def test_tile_differential_diffusion_validation_is_actionable():
    with pytest.raises(ValueError, match="Connect a model"):
        apply_tile_differential_diffusion(None, "core", 1.0)
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        apply_tile_differential_diffusion(MockModel(), "core", 2.0)
    with pytest.raises(ValueError, match="cannot be zero"):
        apply_tile_differential_diffusion(MockModel(), "advanced", 0.0)
