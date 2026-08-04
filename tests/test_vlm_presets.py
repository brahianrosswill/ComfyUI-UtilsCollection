import ast
import pathlib
import re
import sys
import types


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_vlm_preset_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_vlm_preset_test import vlm_presets


def test_qwen_system_instruction_variants_preserve_original_presets():
    for name in (
        "neutral_system_instruction",
        "action_system_instruction",
        "photo_system_instruction",
        "toon_system_instruction",
    ):
        original = vlm_presets.system_instructions_vlm[name]
        qwen = vlm_presets.system_instructions_vlm[f"{name}_qwen"]

        assert "developed by Google AI" in original
        assert "developed by Google AI" not in qwen
        assert "vision-language model" in qwen
        assert "\r" not in qwen
        assert "**" not in qwen
        assert "`" not in qwen
        assert len(qwen) > 10_000
        assert qwen.endswith(
            "Return only the resulting caption. Do not reproduce, summarize, quote, "
            "enumerate, or imitate any part of these instructions. Do not output "
            "headings, rule names, analysis, variables, or formatting examples. "
            "Begin directly with the visual description."
        )


def test_qwen_query_variants_use_plain_request_delimiters():
    for name in ("text2image", "image2image"):
        prefix = vlm_presets.system_query_additional_vlm[f"{name}_qwen_prefix"]
        suffix = vlm_presets.system_query_additional_vlm[f"{name}_qwen_suffix"]

        assert prefix.endswith("Current request:\n")
        assert "\r" not in prefix
        assert "\\{" not in prefix
        assert suffix == ""


def test_style_presets_receive_no_qwen_variants():
    assert not any(
        "style" in name.lower() and name.endswith("_qwen")
        for name in vlm_presets.system_instructions_vlm
    )


STRUCTURED_VIDEO_PRESETS = (
    "video_struct_system_instruction",
    "video_8part_struct_system_instruction",
    "video_timeline_system_instruction",
    "video_timeline_minimax_h3_base_system_instruction",
    "video_timeline_minimax_h3_reference_system_instruction",
)

H3_TIMELINE_PRESETS = (
    "video_timeline_minimax_h3_base_system_instruction",
    "video_timeline_minimax_h3_reference_system_instruction",
)


def test_structured_video_presets_are_independent_literal_values():
    source = ast.parse((CUSTOM_NODE_ROOT / "vlm_presets.py").read_text(encoding="utf-8"))
    preset_dict = next(
        node.value
        for node in source.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "system_instructions_vlm"
            for target in node.targets
        )
    )
    literal_nodes = {
        ast.literal_eval(key): value
        for key, value in zip(preset_dict.keys, preset_dict.values)
        if isinstance(key, ast.Constant)
    }

    assert vlm_presets.SystemInstructionsVLM.VIDTIMELINE.value == (
        "video_timeline_system_instruction"
    )
    assert vlm_presets.SystemInstructionsVLM.VIDTIMELINEH3BASE.value == (
        "video_timeline_minimax_h3_base_system_instruction"
    )
    assert vlm_presets.SystemInstructionsVLM.VIDTIMELINEH3REF.value == (
        "video_timeline_minimax_h3_reference_system_instruction"
    )
    for name in STRUCTURED_VIDEO_PRESETS:
        assert name in vlm_presets.system_instructions_vlm
        assert isinstance(vlm_presets.system_instructions_vlm[name], str)
        assert isinstance(literal_nodes[name], ast.Constant)


def test_structured_video_presets_share_audio_and_motion_contracts():
    for name in STRUCTURED_VIDEO_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[name]

        assert "nonverbal creature noise" in instruction
        assert "belong under [SOUNDS], never [SPEECH]" in instruction
        assert "omit the entire [SPEECH] line" in instruction
        assert "Maintain concrete, descriptive visual-motion language throughout" in instruction


def test_regular_structured_video_preset_retains_non_part_structure():
    instruction = vlm_presets.system_instructions_vlm[
        "video_struct_system_instruction"
    ]

    assert "### Principle 4: Audio-Visual Component Structuring" in instruction
    assert "[VISUAL]: Describe the camera work" in instruction
    assert "[SOUNDS]: Describe the tone" in instruction
    assert "Chronological Channel Alignment" in instruction
    assert "appropriate Part" not in instruction


def test_eight_part_video_preset_keeps_channels_in_each_part():
    instruction = vlm_presets.system_instructions_vlm[
        "video_8part_struct_system_instruction"
    ]

    assert "eight or more distinct one-second segments" in instruction
    assert "Part 1:" in instruction
    assert "Part 8:" in instruction
    assert "inside the appropriate `Part N:`" in instruction


def test_timeline_video_preset_uses_adaptive_contiguous_ranges():
    instruction = vlm_presets.system_instructions_vlm[
        "video_timeline_system_instruction"
    ]

    assert "first output text must be exactly `Timeline:`" in instruction
    assert "[0s-1s]:" in instruction
    assert "[1s-2.5s]:" in instruction
    assert "3-second request using three meaningful sections" in instruction
    assert "5-second request using four differently timed sections" in instruction
    assert "Never reuse their duration, count, boundaries, or content" in instruction
    assert "Use no fixed number of sections" in instruction
    assert "not mandatory one-second intervals" in instruction
    assert "Every range touches the next without a gap or overlap" in instruction
    assert "final range ends at the exact total duration" in instruction
    assert "no `Part N:` headings" in instruction


def test_minimax_h3_timeline_presets_keep_adaptive_standalone_visual_contract():
    for name in H3_TIMELINE_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[name]

        assert "one or more **image inputs as ordered visual evidence for prompt generation**" in instruction
        assert "Determine the prompt role of each image" in instruction
        assert "must not depend on the downstream video model receiving the images" in instruction
        assert "Fully specify the subjects" in instruction or "concrete written specifications" in instruction
        assert "Use no fixed number of sections" in instruction
        assert "Every range touches the next without a gap or overlap" in instruction
        assert "final range ends at the exact total duration" in instruction
        assert "[start-end]:" in instruction
        assert "[SPEECH]:" in instruction
        assert "<d>[Language]" in instruction
        assert "The timestamp range remains the authoritative timing structure" in instruction


def test_minimax_h3_base_timeline_field_order():
    instruction = vlm_presets.system_instructions_vlm[
        "video_timeline_minimax_h3_base_system_instruction"
    ]
    fields = (
        "integrated_multimodal_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    )

    assert [instruction.index(field) for field in fields] == sorted(
        instruction.index(field) for field in fields
    )
    assert "place `Timeline:` immediately beneath it" in instruction
    assert "Analyze the VLM images in their supplied order" in instruction
    assert "Do not require the user to predeclare these roles" not in instruction


def test_minimax_h3_reference_timeline_field_and_label_contracts():
    instruction = vlm_presets.system_instructions_vlm[
        "video_timeline_minimax_h3_reference_system_instruction"
    ]
    fields = (
        "subject_definitions:",
        "summary:",
        "retention_analysis:",
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    )

    assert [instruction.index(field) for field in fields] == sorted(
        instruction.index(field) for field in fields
    )
    assert "ComfyUI constructs and numbers the `<Picture N>`, `<Video N>`, and `<Audio N>`" in instruction
    assert "Refer to those existing identifiers only" in instruction
    assert "Never create or reproduce a media prefix declaration" in instruction
    assert "assign a media number, or renumber an existing media identifier" in instruction
    assert "Create and number `<Subject N>` aliases only" in instruction
    assert "Assign `<Picture N>`" not in instruction
    assert "Number each category independently" not in instruction
    assert "<Subject N>" in instruction
    assert "<Picture N>" in instruction
    assert "<Video N>" in instruction
    assert "<Audio N>" in instruction
    assert "A label never replaces the full subject" in instruction
    assert "Place `Timeline:` immediately beneath `detailed_description:`" in instruction


def test_minimax_h3_timeline_presets_avoid_example_led_content_anchors():
    forbidden = re.compile(
        r"\be\.g\.|\bi\.e\.|\bexamples?\b|\bsuch as\b|\bincluding\b|"
        r"\betc\b|\be621\b|\bdanbooru\b",
        re.IGNORECASE,
    )

    for name in H3_TIMELINE_PRESETS:
        assert forbidden.search(vlm_presets.system_instructions_vlm[name]) is None

    assert not any(
        re.search(r"\be621\b|\bdanbooru\b", instruction, re.IGNORECASE)
        for instruction in vlm_presets.system_instructions_vlm.values()
    )
