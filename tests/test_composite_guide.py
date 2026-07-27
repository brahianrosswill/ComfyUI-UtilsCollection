import pathlib
import sys
import types


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_composite_guide_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_composite_guide_test.utils_nodes import (
        UC_CompositeNodesGuide,
    )
finally:
    cli_args.cpu = prior_cpu


EXPECTED_TOPICS = [
    "node_catalog",
    "model_inputs",
    "mask_cleanup_and_resize",
    "unified_background_replace",
    "layered_composite",
    "staged_workflow",
    "staged_face_workflow",
    "placement_editor",
    "mediapipe_face_composite",
]


def test_composite_guide_topics_render_markdown():
    topic_input = {
        value.id: value for value in UC_CompositeNodesGuide.define_schema().inputs
    }["topic"]

    assert topic_input.options == EXPECTED_TOPICS
    assert topic_input.default == "node_catalog"
    for topic in EXPECTED_TOPICS:
        markdown = UC_CompositeNodesGuide.execute(topic).args[0]
        assert markdown.startswith("## ")
        assert "Unknown topic" not in markdown


def test_composite_guide_documents_optional_model_contract():
    markdown = UC_CompositeNodesGuide.execute("model_inputs").args[0]

    assert "`background_removal_model_opt`" in markdown
    assert "`face_detection_model_opt`" in markdown
    assert "`birefnet.safetensors`" in markdown
    assert "`mediapipe_face_fp32.safetensors`" in markdown
    assert "connected Lucida model" in markdown


def test_composite_guide_documents_staged_modes_and_face_defaults():
    staged = UC_CompositeNodesGuide.execute("staged_workflow").args[0]
    face = UC_CompositeNodesGuide.execute("staged_face_workflow").args[0]

    assert all(mode in staged for mode in ("`run_staging`", "`run_staged`", "`full_run`"))
    assert "`run_staged`" in face
    assert "`detection_threshold=0.55`" in face
    assert "`maximum_faces=16`" in face
