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
