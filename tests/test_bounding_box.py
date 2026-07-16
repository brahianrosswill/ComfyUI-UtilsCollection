import pathlib
import sys
import types

import pytest


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_bbox_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_bbox_test.utils_nodes import UC_AdjustBoundingBox


def _result(data, index=0, expansion=0, axis="both", multiple="0"):
    return UC_AdjustBoundingBox.execute(data, index, expansion, axis, multiple).args[0]


def test_extract_and_uniform_expansion():
    data = {"detections": [{"x": 10, "y": 20, "width": 30, "height": 40}]}
    assert _result(data, expansion=5) == {"x": 5, "y": 15, "width": 40, "height": 50}


def test_axis_expansion_and_multiple_alignment():
    data = [{"x": 100, "y": 200, "width": 31, "height": 33}]
    assert _result(data, expansion=3, axis="horizontal", multiple="16") == {
        "x": 92,
        "y": 193,
        "width": 48,
        "height": 48,
    }


def test_index_selection_and_json_input():
    data = '[{"x": 1, "y": 2, "width": 8, "height": 9}, {"x": 4, "y": 5, "width": 6, "height": 7}]'
    assert _result(data, index=1) == {"x": 4, "y": 5, "width": 6, "height": 7}


def test_invalid_box_rejected():
    with pytest.raises(ValueError, match="positive width and height"):
        _result({"x": 0, "y": 0, "width": 0, "height": 8})
