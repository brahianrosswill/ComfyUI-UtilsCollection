import pathlib
import sys
import types

import numpy as np
import pytest
import torch


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_composite_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_composite_test import composite_nodes
finally:
    cli_args.cpu = prior_cpu


def test_resize_mask_preserves_asymmetric_orientation():
    mask = torch.zeros(1, 2, 3)
    mask[0, 0, 0] = 1.0

    output = composite_nodes.UC_ResizeMask.execute(mask, 6, 4, False, "nearest-exact", "disabled")
    resized, width, height = output.result

    assert (width, height) == (6, 4)
    assert torch.equal(resized[0, :2, :2], torch.ones(2, 2))
    assert resized[0, 2:, :].sum() == 0
    assert resized[0, :, 2:].sum() == 0


def test_image_and_mask_resize_uses_target_and_optional_overrides():
    image = torch.zeros(1, 2, 3, 3)
    image[0, 0, 0, 0] = 1.0
    mask = image[..., 0]
    target = torch.zeros(1, 8, 10, 3)

    target_sized = composite_nodes.UC_ImageAndMaskResize.execute(image, mask, target, "nearest-exact", "disabled", 0)
    width_overridden = composite_nodes.UC_ImageAndMaskResize.execute(image, mask, target, "nearest-exact", "disabled", 0, width=6)

    assert target_sized.result[0].shape == (1, 8, 10, 3)
    assert target_sized.result[1].shape == (1, 8, 10)
    assert width_overridden.result[0].shape == (1, 8, 6, 3)
    assert width_overridden.result[1].shape == (1, 8, 6)
    assert target_sized.result[0][0, :4, :4, 0].sum() > 0
    assert target_sized.result[0][0, 4:, :, 0].sum() == 0


def test_crop_by_mask_uses_batch_union_and_rejects_empty_masks():
    image = torch.zeros(2, 32, 32, 3)
    mask = torch.zeros(2, 32, 32)
    mask[0, 4:6, 4:6] = 1.0
    mask[1, 24:26, 24:26] = 1.0

    output = composite_nodes.UC_CropByMask.execute(image, mask, 0)

    assert output.result[0].shape[0] == 2
    cropped_mask, crop_x, crop_y = output.result[1:4]
    assert cropped_mask[0, 4 - crop_y:6 - crop_y, 4 - crop_x:6 - crop_x].sum() == 4
    assert cropped_mask[1, 24 - crop_y:26 - crop_y, 24 - crop_x:26 - crop_x].sum() == 4
    with pytest.raises(ValueError, match="Mask is empty"):
        composite_nodes.UC_CropByMask.execute(image[:1], torch.zeros(1, 32, 32), 0)


def test_crop_by_mask_exposes_dimension_multiple_without_resizing():
    image = torch.arange(64 * 64 * 3, dtype=torch.float32).reshape(1, 64, 64, 3)
    mask = torch.zeros(1, 64, 64)
    mask[:, 20:29, 22:31] = 1.0

    output = composite_nodes.UC_CropByMask.execute(image, mask, padding=2, multiple=16)
    cropped_image, cropped_mask, crop_x, crop_y, width, height = output.result

    assert (width, height) == (16, 16)
    assert cropped_image.shape == (1, 16, 16, 3)
    assert cropped_mask.shape == (1, 16, 16)
    assert torch.equal(cropped_image, image[:, crop_y:crop_y + height, crop_x:crop_x + width])
    assert cropped_mask.sum() == 81


def test_crop_by_mask_multiple_defaults_to_legacy_eight_pixels():
    schema = composite_nodes.UC_CropByMask.define_schema()
    multiple = next(value for value in schema.inputs if value.id == "multiple")
    mask = torch.zeros(1, 64, 64)
    mask[:, 20:29, 22:31] = 1.0

    output = composite_nodes.UC_CropByMask.execute(torch.zeros(1, 64, 64, 3), mask, 2)

    assert multiple.default == 8
    assert multiple.min == 4
    assert multiple.step == 4
    assert output.result[4:] == (16, 16)


def test_crop_merge_supports_mask_and_singleton_broadcast():
    original = torch.zeros(2, 8, 8, 3)
    crop = torch.ones(1, 4, 4, 3)
    mask = torch.zeros(1, 4, 4)
    mask[:, :, :2] = 1.0

    output = composite_nodes.UC_ImageCropMerge.execute(crop, original, 2, 2, 4, 4, "nearest-exact", mask).result[0]

    assert torch.equal(output[:, 2:6, 2:4], torch.ones(2, 4, 2, 3))
    assert output[:, 2:6, 4:6].sum() == 0


def test_mask_expansion_and_feather_support_contraction():
    mask = torch.zeros(9, 9)
    mask[2:7, 2:7] = 1.0

    expanded = composite_nodes._expand_mask(mask, 1)
    contracted = composite_nodes._expand_mask(mask, -1)
    outward = composite_nodes._feather_mask(mask, 2)
    inward = composite_nodes._feather_mask(mask, -2)

    assert expanded.sum() > mask.sum()
    assert contracted.sum() < mask.sum()
    assert outward.sum() > mask.sum()
    assert inward.sum() < mask.sum()
    assert torch.equal(outward[2:7, 2:7], mask[2:7, 2:7])
    assert inward[:2].sum() == 0
    assert inward[:, :2].sum() == 0


class _QueuedBackgroundModel:
    def __init__(self, masks):
        self.masks = list(masks)
        self.colors = []

    def encode_image(self, image):
        self.colors.append(image[0, 0, 0].clone())
        return self.masks.pop(0).to(image)


def _replace_background(model, background, foregrounds, **overrides):
    options = {
        "foreground_scale": 0.9,
        "long_axis_shift": 0.0,
        "short_axis_shift": 0.0,
        "mask_threshold": 0.5,
        "border_cleanup_width": 0,
        "artifact_cleanup_radius": 0,
        "gap_fill_radius": 0,
        "feather_radius": 0,
    }
    options.update(overrides)
    return composite_nodes.UC_UnifiedBackgroundReplace.execute(
        model, background, foregrounds, **options
    ).result


def test_unified_background_flattens_inputs_and_centers_foreground_bounds():
    background = torch.zeros(1, 100, 160, 3, dtype=torch.float64)
    first = torch.zeros(1, 10, 10, 3)
    first[..., 0] = 1.0
    second = torch.zeros(2, 8, 12, 3)
    second[0, ..., 1] = 1.0
    second[1, ..., 2] = 1.0
    square_mask = torch.zeros(1, 5, 5)
    square_mask[:, 1:4, 1:4] = 1.0
    wide_mask = torch.zeros(1, 8, 12)
    wide_mask[:, 2:6, 2:10] = 1.0
    tall_mask = torch.zeros(1, 8, 12)
    tall_mask[:, :, 4:8] = 1.0
    model = _QueuedBackgroundModel([square_mask, wide_mask, tall_mask])

    images, masks = _replace_background(
        model,
        background,
        {"foreground_10": second, "foreground_2": first},
    )

    assert images.shape == (3, 100, 160, 3)
    assert masks.shape == (3, 100, 160)
    assert images.dtype == background.dtype
    assert images.device == background.device
    assert [color.argmax().item() for color in model.colors] == [0, 1, 2]
    assert set(masks.unique().tolist()) == {0.0, 1.0}
    assert masks[0].sum() == 90 * 90
    assert masks[1].sum() == 45 * 90
    assert masks[2].sum() == 45 * 90
    assert masks[0, 5:95, 35:125].all()
    assert masks[1, 28:73, 35:125].all()
    assert masks[2, 5:95, 58:103].all()


def test_unified_background_refines_weak_edges_gaps_and_artifacts():
    raw = torch.zeros(12, 12)
    raw[3:10, 3:10] = 0.9
    raw[6, 6] = 0.0
    raw[0, 0:3] = 0.6
    raw[1, 11] = 0.9

    refined = composite_nodes._refine_foreground_mask(raw, 0.5, 2, 1, 1)

    assert refined[0, 0:3].sum() == 0
    assert refined[1, 11] == 0
    assert refined[6, 6] == 1
    assert refined[4:9, 4:9].all()


def test_unified_background_keeps_strong_edge_subject_and_feathers_only_boundary():
    background = torch.zeros(1, 64, 80, 3)
    foreground = torch.ones(1, 16, 16, 3)
    raw = torch.zeros(1, 16, 16)
    raw[:, 0:14, 3:13] = 0.95
    model = _QueuedBackgroundModel([raw])

    _, masks = _replace_background(
        model,
        background,
        {"foreground_0": foreground},
        border_cleanup_width=2,
        feather_radius=2,
    )

    assert masks.max() == 1
    assert masks.min() == 0
    assert ((masks > 0) & (masks < 1)).any()
    assert masks[0, 32, 40] == 1


@pytest.mark.parametrize(
    ("background_shape", "shift", "expected_bounds"),
    [
        ((60, 100), -1.0, (3, 57, 0, 54)),
        ((60, 100), 1.0, (3, 57, 46, 100)),
        ((100, 60), -1.0, (0, 54, 3, 57)),
        ((100, 60), 1.0, (46, 100, 3, 57)),
    ],
)
def test_unified_background_shifts_along_background_long_axis(background_shape, shift, expected_bounds):
    height, width = background_shape
    background = torch.zeros(1, height, width, 3)
    foreground = torch.ones(1, 8, 8, 3)
    model = _QueuedBackgroundModel([torch.ones(1, 8, 8)])

    _, masks = _replace_background(
        model,
        background,
        {"foreground_0": foreground},
        long_axis_shift=shift,
    )

    top, bottom, left, right = expected_bounds
    assert masks[0, top:bottom, left:right].all()
    assert masks.sum() == (bottom - top) * (right - left)


def test_unified_background_overscale_crops_to_canvas_perimeter():
    background = torch.zeros(1, 40, 60, 3)
    foreground = torch.ones(1, 8, 8, 3)
    model = _QueuedBackgroundModel([torch.ones(1, 8, 8)])

    images, masks = _replace_background(
        model,
        background,
        {"foreground_0": foreground},
        foreground_scale=2.0,
    )

    assert images.shape == background.shape
    assert masks.shape == background.shape[:3]
    assert masks.all()
    assert images.all()


@pytest.mark.parametrize(
    ("background_shape", "shift", "expected_bounds"),
    [
        ((60, 100), -1.0, (0, 54, 23, 77)),
        ((60, 100), 1.0, (6, 60, 23, 77)),
        ((100, 60), -1.0, (23, 77, 0, 54)),
        ((100, 60), 1.0, (23, 77, 6, 60)),
    ],
)
def test_unified_background_shifts_along_background_short_axis(background_shape, shift, expected_bounds):
    height, width = background_shape
    background = torch.zeros(1, height, width, 3)
    foreground = torch.ones(1, 8, 8, 3)
    model = _QueuedBackgroundModel([torch.ones(1, 8, 8)])

    _, masks = _replace_background(
        model,
        background,
        {"foreground_0": foreground},
        short_axis_shift=shift,
    )

    top, bottom, left, right = expected_bounds
    assert masks[0, top:bottom, left:right].all()
    assert masks.sum() == (bottom - top) * (right - left)


def test_unified_background_square_canvas_uses_both_axis_shifts():
    background = torch.zeros(1, 100, 100, 3)
    foreground = torch.ones(1, 8, 8, 3)
    model = _QueuedBackgroundModel([torch.ones(1, 8, 8)])

    _, masks = _replace_background(
        model,
        background,
        {"foreground_0": foreground},
        long_axis_shift=-1.0,
        short_axis_shift=1.0,
    )

    assert masks[0, 10:100, 0:90].all()
    assert masks.sum() == 90 * 90


def test_unified_background_validates_background_and_empty_masks():
    foreground = torch.ones(1, 8, 8, 3)
    model = _QueuedBackgroundModel([torch.zeros(1, 4, 4)])
    with pytest.raises(ValueError, match="exactly one background"):
        _replace_background(model, torch.zeros(2, 16, 16, 3), {"foreground_0": foreground})
    with pytest.raises(ValueError, match="empty foreground mask for image 1"):
        _replace_background(model, torch.zeros(1, 16, 16, 3), {"foreground_0": foreground})


def _layered_composite(model, background, foregrounds, placement_data=None, **overrides):
    options = {
        "placement_data": placement_data or '{"version":1,"layers":{}}',
        "mask_threshold": 0.5,
        "border_cleanup_width": 0,
        "artifact_cleanup_radius": 0,
        "gap_fill_radius": 0,
        "feather_radius": 0,
    }
    options.update(overrides)
    return composite_nodes.UC_LayeredBackgroundComposite.execute(
        model, background, foregrounds, **options
    )


def test_layered_background_composites_in_socket_order(monkeypatch):
    monkeypatch.setattr(composite_nodes, "_save_editor_preview", lambda image, prefix, longest: {"filename": prefix})
    background = torch.zeros(1, 20, 20, 3)
    red = torch.zeros(1, 4, 4, 3)
    red[..., 0] = 1
    green = torch.zeros(1, 4, 4, 3)
    green[..., 1] = 1
    model = _QueuedBackgroundModel([torch.ones(1, 4, 4), torch.ones(1, 4, 4)])

    output = _layered_composite(
        model,
        background,
        {"foreground_1": green, "foreground_0": red},
        '{"version":1,"layers":{"foreground_0":{"scale":0.5},"foreground_1":{"scale":0.25}}}',
    )
    image, mask = output.result

    assert image.shape == (1, 20, 20, 3)
    assert mask.shape == (1, 20, 20)
    assert torch.allclose(image[0, 8:13, 8:13, 1], torch.ones(5, 5))
    assert image[0, 5:15, 5:15, 0].sum().item() == pytest.approx(75)
    assert mask.sum() == 100
    metadata = output.ui["uc_layered_scene_editor"][0]
    assert [(layer["socket"], layer["crop_width"], layer["crop_height"]) for layer in metadata["layers"]] == [
        ("foreground_0", 4, 4),
        ("foreground_1", 4, 4),
    ]


def test_layered_background_uses_independent_landscape_positions(monkeypatch):
    monkeypatch.setattr(composite_nodes, "_save_editor_preview", lambda *args: None)
    background = torch.zeros(1, 20, 40, 3)
    left = torch.ones(1, 4, 4, 3)
    right = torch.ones(1, 4, 4, 3) * 0.5
    model = _QueuedBackgroundModel([torch.ones(1, 4, 4), torch.ones(1, 4, 4)])
    placement = (
        '{"version":1,"layers":{'
        '"foreground_0":{"scale":0.5,"long_axis_shift":-1,"short_axis_shift":0},'
        '"foreground_1":{"scale":0.5,"long_axis_shift":1,"short_axis_shift":0}}}'
    )

    image, mask = _layered_composite(
        model, background, {"foreground_0": left, "foreground_1": right}, placement
    ).result

    assert torch.allclose(image[0, 5:15, 0:10], torch.ones(10, 10, 3))
    assert torch.allclose(image[0, 5:15, 30:40], torch.full((10, 10, 3), 0.5))
    assert mask.sum() == 200


def test_layered_background_uses_explicit_layer_order(monkeypatch):
    monkeypatch.setattr(composite_nodes, "_save_editor_preview", lambda *args: None)
    background = torch.zeros(1, 20, 20, 3)
    red = torch.zeros(1, 4, 4, 3)
    red[..., 0] = 1
    green = torch.zeros(1, 4, 4, 3)
    green[..., 1] = 1
    model = _QueuedBackgroundModel([torch.ones(1, 4, 4), torch.ones(1, 4, 4)])
    placement = (
        '{"version":1,"layer_order":["foreground_1","foreground_0"],"layers":{'
        '"foreground_0":{"scale":0.5},"foreground_1":{"scale":0.5}}}'
    )

    image, _ = _layered_composite(
        model, background, {"foreground_0": red, "foreground_1": green}, placement
    ).result

    assert torch.allclose(image[0, 5:15, 5:15, 0], torch.ones(10, 10))
    assert image[0, 5:15, 5:15, 1].sum().item() == 0


def test_layered_background_rejects_batches_and_invalid_placements(monkeypatch):
    monkeypatch.setattr(composite_nodes, "_save_editor_preview", lambda *args: None)
    background = torch.zeros(1, 16, 16, 3)
    foreground = torch.ones(2, 4, 4, 3)
    with pytest.raises(ValueError, match="not a batch"):
        _layered_composite(
            _QueuedBackgroundModel([]), background, {"foreground_0": foreground}
        )
    with pytest.raises(ValueError, match="scale must be between"):
        _layered_composite(
            _QueuedBackgroundModel([]),
            background,
            {"foreground_0": foreground[:1]},
            '{"version":1,"layers":{"foreground_0":{"scale":20}}}',
        )


def test_layered_background_preview_failure_does_not_change_result(monkeypatch):
    def fail_preview(*args):
        raise OSError("preview unavailable")

    monkeypatch.setattr(composite_nodes, "_save_editor_preview", fail_preview)
    background = torch.zeros(1, 12, 12, 3)
    foreground = torch.ones(1, 4, 2, 3)
    model = _QueuedBackgroundModel([torch.ones(1, 4, 2)])

    output = _layered_composite(model, background, {"foreground_0": foreground})

    assert output.result[0].max().item() == pytest.approx(1)
    metadata = output.ui["uc_layered_scene_editor"][0]
    assert "preview" not in metadata["background"]
    assert "preview" not in metadata["layers"][0]
    assert metadata["layers"][0]["crop_width"] == 2
    assert metadata["layers"][0]["crop_height"] == 4


def test_staged_layered_composite_reuses_prepared_cutouts(monkeypatch):
    monkeypatch.setattr(composite_nodes, "_save_editor_preview", lambda image, prefix, longest: {"filename": prefix})
    foreground = torch.zeros(1, 6, 8, 3)
    foreground[..., 0] = 1
    mask = torch.zeros(1, 6, 8)
    mask[:, 1:5, 2:6] = 1
    model = _QueuedBackgroundModel([mask])

    staged = composite_nodes._stage_layered_foregrounds(
        model,
        {"foreground_0": foreground},
        0.5,
        0,
        0,
        0,
    )
    output = composite_nodes._composite_staged_foregrounds(
        torch.zeros(1, 20, 20, 3),
        staged,
        '{"version":1,"layers":{"foreground_0":{"scale":0.5}}}',
        0,
    )
    image, placed_mask = output.result

    assert len(model.masks) == 0
    assert staged["layers"][0]["image"].shape[1:3] == (4, 4)
    assert image[..., 0].sum().item() == pytest.approx(100)
    assert placed_mask.sum().item() == pytest.approx(100)
    assert output.ui["uc_layered_scene_editor"][0]["layers"][0]["crop_width"] == 4


def test_staged_layered_composite_rejects_missing_stage():
    with pytest.raises(ValueError, match="missing or incompatible"):
        composite_nodes._composite_staged_foregrounds(
            torch.zeros(1, 20, 20, 3), None, '{"version":1,"layers":{}}', 0
        )


def test_staged_compositor_publishes_transparent_cutout_previews(monkeypatch):
    saved = []

    def capture_preview(image, prefix, longest):
        saved.append(image.clone())
        return {"filename": f"{prefix}.png"}

    monkeypatch.setattr(composite_nodes, "_save_editor_preview", capture_preview)
    node = composite_nodes.UC_StagedLayeredBackgroundComposite
    node._staged_by_node.clear()
    monkeypatch.setattr(node, "hidden", types.SimpleNamespace(unique_id="preview-compositor"))
    foreground = torch.ones(1, 6, 8, 3)
    mask = torch.zeros(1, 6, 8)
    mask[:, 1:5, 3] = 1
    mask[:, 3, 2:6] = 1
    background = torch.zeros(1, 12, 12, 3)
    output = node.execute(
        _QueuedBackgroundModel([mask]),
        background,
        {"foreground_0": foreground},
        False,
        0.5,
        0,
        0,
        0,
        '{"version":1,"layers":{}}',
        0,
    )

    assert torch.equal(output.result[0], background)
    assert output.result[1].sum().item() == 0
    assert output.ui["uc_layered_scene_editor"][0]["stage_mode"] == "fresh"
    assert saved[1].shape == (1, 4, 4, 4)
    assert saved[1][..., 3].min().item() == 0
    assert saved[1][..., 3].max().item() == 1


def test_staged_compositor_lazily_resumes_its_own_stage(monkeypatch):
    node = composite_nodes.UC_StagedLayeredBackgroundComposite
    schema = node.define_schema()
    model_input = next(value for value in schema.inputs if value.id == "background_removal_model")
    foreground_input = next(value for value in schema.inputs if value.id == "foreground_images")
    assert schema.is_output_node is True
    assert model_input.lazy is True
    assert foreground_input.template.input.lazy is True
    node._staged_by_node.clear()
    monkeypatch.setattr(node, "hidden", types.SimpleNamespace(unique_id="compositor-a"))
    monkeypatch.setattr(composite_nodes, "_save_editor_preview", lambda *args: None)
    background = torch.zeros(1, 20, 20, 3)

    model = _QueuedBackgroundModel([torch.ones(1, 4, 4)])
    foregrounds = {"foreground_0": torch.ones(1, 4, 4, 3)}
    assert node.check_lazy_status(
        False, None, {"foreground_0": (None, "foreground_0")}
    ) == ["background_removal_model", "foreground_0"]
    assert node.check_lazy_status(
        False, model, {"foreground_0": (foregrounds["foreground_0"], "foreground_0")}
    ) == []
    assert node.check_lazy_status(True, None, {"foreground_0": (None, "foreground_0")}) == []
    fresh = node.execute(
        model, background, foregrounds, False, 0.5, 0, 0, 0,
        '{"version":1,"layers":{}}', 0,
    )
    retained = node.execute(
        None, background, {"foreground_0": None}, True, 0.5, 0, 0, 0,
        '{"version":1,"layers":{}}', 0,
    )

    assert fresh.ui["uc_layered_scene_editor"][0]["stage_mode"] == "fresh"
    assert retained.ui["uc_layered_scene_editor"][0]["stage_mode"] == "retained"
    assert fresh.result[0].sum().item() == 0
    assert retained.result[0].sum().item() > 0

    monkeypatch.setattr(node, "hidden", types.SimpleNamespace(unique_id="compositor-b"))
    with pytest.raises(ValueError, match="No retained foreground stage"):
        node.execute(
            None, background, {"foreground_0": None}, True, 0.5, 0, 0, 0,
            '{"version":1,"layers":{}}', 0,
        )


def test_face_foreground_solidification_matches_composite_operation():
    foreground = torch.tensor([0.25, 0.5, 0.625, 0.75, 1.0])
    inverted = 1.0 - foreground
    solid = ((foreground - inverted) * 2.0).clamp(0.0, 1.0)

    assert torch.equal(solid, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))


def test_target_warp_keeps_crop_border_fixed():
    y, x = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    target = torch.stack((x, y, x + y), dim=-1).to(torch.float32).div_(30.0)
    source_points = np.array([[5, 5], [11, 5], [11, 11], [5, 11]], dtype=np.float32)
    target_points = source_points + np.array([2, 0], dtype=np.float32)

    warped = composite_nodes._warp_target(target, source_points, target_points, 1.0, 4)

    assert torch.allclose(warped[0], target[0], atol=1e-4)
    assert torch.allclose(warped[-1], target[-1], atol=1e-4)
    assert torch.allclose(warped[:, 0], target[:, 0], atol=1e-4)
    assert torch.allclose(warped[:, -1], target[:, -1], atol=1e-4)
    assert not torch.allclose(warped[5:12, 5:12], target[5:12, 5:12])
    for source_point, target_point in zip(source_points.astype(int), target_points.astype(int)):
        expected = target[target_point[1], target_point[0]]
        actual = warped[source_point[1], source_point[0]]
        assert torch.allclose(actual, expected, atol=2e-3)


def test_similarity_transform_allows_rotation_without_source_warp():
    source = np.array([[2, 2], [8, 2], [8, 10], [2, 10]], dtype=np.float32)
    rotation = np.array([[0, -1], [1, 0]], dtype=np.float32)
    target = 1.5 * (source @ rotation.T) + np.array([20, 5], dtype=np.float32)

    scale, solved_rotation, translation = composite_nodes._similarity_transform(source, target)
    transformed = scale * (source @ solved_rotation.T) + translation

    assert scale == pytest.approx(1.5)
    assert np.allclose(solved_rotation, rotation, atol=1e-6)
    assert np.allclose(transformed, target, atol=1e-5)


class _FaceModel:
    def __init__(self):
        self.connection_sets = {"face_oval": frozenset({(0, 1), (1, 2), (2, 3), (3, 0)})}
        self.calls = []

    def detect_batch(self, images, num_faces, score_thresh, variant):
        self.calls.append((images[0].shape, num_faces, score_thresh, variant))
        height, width = images[0].shape[:2]
        if width == 20:
            landmarks = np.array([[4, 8], [8, 4], [12, 8], [8, 12]], dtype=np.float32)
            box = np.array([4, 4, 12, 12], dtype=np.float32)
        else:
            landmarks = np.array([[10, 14], [16, 8], [22, 14], [16, 20]], dtype=np.float32)
            box = np.array([10, 8, 22, 20], dtype=np.float32)
        smaller = {"bbox_xyxy": box * 0.5, "landmarks_xy": landmarks * 0.5}
        larger = {"bbox_xyxy": box, "landmarks_xy": landmarks}
        return [[smaller, larger]]


class _BackgroundModel:
    def encode_image(self, image):
        return torch.ones(image.shape[0], image.shape[1], image.shape[2], device=image.device, dtype=image.dtype)


def test_face_composite_uses_full_detector_and_target_crop_coordinates():
    source = torch.zeros(1, 20, 20, 3)
    source[..., 0] = 1.0
    target = torch.zeros(1, 30, 30, 3)
    face_model = _FaceModel()
    options = {
        "bbox_expansion": 2,
        "mask_expansion": 0,
        "feather_radius": 0,
        "target_warp_strength": 0.0,
        "warp_decay_radius": 4,
    }

    output = composite_nodes.UC_MediaPipeFaceComposite.execute(face_model, _BackgroundModel(), source, target, options)
    image, crop = output.result

    assert [call[3] for call in face_model.calls] == ["full", "full"]
    assert crop.shape == (1, 16, 16, 3)
    assert image[0, 6:22, 8:24, 0].sum() > 0
    assert image[0, :6].sum() == 0
    assert image[0, :, :8].sum() == 0


def test_face_composite_rejects_batches_and_missing_faces():
    source = torch.zeros(2, 20, 20, 3)
    target = torch.zeros(1, 30, 30, 3)
    with pytest.raises(ValueError, match="requires one source"):
        composite_nodes.UC_MediaPipeFaceComposite.execute(_FaceModel(), _BackgroundModel(), source, target)

    model = _FaceModel()
    model.detect_batch = lambda *args, **kwargs: [[]]
    with pytest.raises(ValueError, match="No face was detected"):
        composite_nodes.UC_MediaPipeFaceComposite.execute(model, _BackgroundModel(), source[:1], target)
