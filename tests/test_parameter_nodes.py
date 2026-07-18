import pathlib
import sys
import types

import torch


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_parameter_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_parameter_test import parameter_nodes
    from utils_collection_parameter_test.helper_functions import AspectRatio
finally:
    cli_args.cpu = prior_cpu


def test_image_scale_picker_schema_uses_smooth_default_and_positive_scale():
    schema = parameter_nodes.UC_ImageScaleAndResolutionPicker.define_schema()
    inputs = {value.id: value for value in schema.inputs}

    assert inputs["upscale_method"].default == "lanczos"
    assert inputs["scale_by"].min > 0


def test_image_scale_picker_keeps_megapixel_fit_and_upscale_factor_separate():
    image = torch.zeros(1, 100, 200, 3)

    output = parameter_nodes.UC_ImageScaleAndResolutionPicker.execute(
        image=image,
        upscale_method="bilinear",
        crop_method="disabled",
        aspect_ratio=AspectRatio.SQUARE,
        megapixels=0.01,
        resolution_steps=256,
        scale_by=2.0,
        multiple=16,
    )
    adjusted, upscaled, width, height, upscaled_width, upscaled_height = output.result

    assert (width, height) == (144, 80)
    assert (upscaled_width, upscaled_height) == (288, 160)
    assert adjusted.shape == (1, 80, 144, 3)
    assert upscaled.shape == (1, 160, 288, 3)


def test_image_scale_picker_center_crop_uses_adjusted_base_for_upscale():
    image = torch.zeros(1, 100, 200, 3)

    output = parameter_nodes.UC_ImageScaleAndResolutionPicker.execute(
        image=image,
        upscale_method="bilinear",
        crop_method="center",
        aspect_ratio=AspectRatio.SQUARE,
        megapixels=0.01,
        resolution_steps=1,
        scale_by=1.5,
        multiple=16,
    )

    assert output.result[2:] == (96, 96, 144, 144)
    assert output.result[0].shape == (1, 96, 96, 3)
    assert output.result[1].shape == (1, 144, 144, 3)
