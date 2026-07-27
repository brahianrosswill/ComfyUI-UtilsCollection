import pathlib
import sys
import types


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_encoder_guide_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_encoder_guide_test.utils_nodes import UC_EncoderNodesGuide
finally:
    cli_args.cpu = prior_cpu


EXPECTED_TOPICS = [
    "node_catalog",
    "prompt_templates_and_weighting",
    "image_inputs_and_placeholders",
    "resolution_and_reference_latents",
    "visual_fusion",
    "consensus_settings",
    "formulas_and_alignment",
    "embedding_export",
    "compatibility_nodes",
]


def test_encoder_guide_topics_are_complete_and_render_markdown():
    topic_input = {
        value.id: value for value in UC_EncoderNodesGuide.define_schema().inputs
    }["topic"]

    assert topic_input.options == EXPECTED_TOPICS
    assert topic_input.default == "node_catalog"
    for topic in EXPECTED_TOPICS:
        markdown = UC_EncoderNodesGuide.execute(topic).args[0]
        assert markdown.startswith("## ")
        assert "Unknown topic" not in markdown


def test_encoder_guide_covers_current_visual_fusion_contract():
    resolution = UC_EncoderNodesGuide.execute(
        "resolution_and_reference_latents"
    ).args[0]
    fusion = UC_EncoderNodesGuide.execute("visual_fusion").args[0]
    consensus = UC_EncoderNodesGuide.execute("consensus_settings").args[0]

    assert all(value in resolution for value in ("256", "3584", "step `2`", "1` through `15"))
    assert all(
        value in fusion
        for value in (
            "UC_AdvancedVisualConditioningEncode",
            "UC_Krea2TokenAttentionWeight",
            "spatial-checkerboard",
            "spatial-block-interleave",
            "spatial-dither-random",
            "`fusion_strength=1.0`",
            "`0.0` uses only consensus weights",
        )
    )
    assert "Text Scaled Encoder (Advanced)" not in fusion
    assert "aligned by grid coordinate" in consensus
    assert "common-prefix matching are not applied to the visual span" in consensus


def test_encoder_guide_labels_compatibility_nodes_separately():
    catalog = UC_EncoderNodesGuide.execute("node_catalog").args[0]
    compatibility = UC_EncoderNodesGuide.execute("compatibility_nodes").args[0]

    assert "`UC_AdvancedVisualConditioningEncode`" in catalog
    assert "`UC_VLMInputEmbeds`" in catalog
    assert "`UC_Krea2TokenAttentionWeight`" in catalog
    assert "`TextEncodeKrea2SystemEditScaledAdv`" in compatibility
    assert "`UC_Qwen3VLInputEmbeds`" in compatibility
