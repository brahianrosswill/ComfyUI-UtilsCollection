import ast
import hashlib
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


HARDENED_IMAGE_PRESET_BASELINES = {
    "neutral_system_instruction": (
        16_397,
        2_353,
        "bcf02ccfda454c5e0869eb5976311d45e0b574fb2d2140f65f4681755c88e4b6",
    ),
    "action_system_instruction": (
        16_662,
        2_375,
        "01349683d901dcc9e8055272b4377fb4cebef2cba40d5c1cbdbf043c4db74905",
    ),
    "photo_system_instruction": (
        17_229,
        2_499,
        "d339359f274b4bb9e09d4df756b2ad1c8055772c3fa260e8173ba01b7d65203d",
    ),
    "toon_system_instruction": (
        18_919,
        2_756,
        "f924fb6a51b2b828eb54937493309f4301dd38faa56eda923f13bf3b04ef0caf",
    ),
    "neutral_system_instruction_crude": (
        16_962,
        2_412,
        "a1f63e248dfabc277f8be70380d08d9d00c94b4a220fef6820b4d9f1c232ee62",
    ),
    "action_system_instruction_crude": (
        16_679,
        2_361,
        "5a3ac54c9977cce48981971dc8dcf4b979c60f3d41e075fb528ea76b95f364ee",
    ),
    "photo_system_instruction_crude": (
        18_280,
        2_629,
        "7848c57f4c98da5777d7a833bd419fc03d600400cd423e284ed5df2d3acd6f59",
    ),
    "toon_system_instruction_crude": (
        9_876,
        1_349,
        "0adfb0dfd846bda93ddc847177ce9a39b402788d1160d600a0cccb6ce6f982bd",
    ),
}
HARDENING_HEADINGS = (
    "## Perspective and Spatial Description",
    "## Visible Text Quotation",
    "## Direct Language Constraints",
)
HARDENING_FORBIDDEN = re.compile(
    r"\be\.g\.\b|\bi\.e\.\b|\bexamples?\b|\bfor instance\b|"
    r"\bsuch as\b|\bincluding\b|\bsample outputs?\b|\bphrase menus?\b|"
    r"\bchoose from\b|\bpossible (?:terms|phrases|values)\b",
    re.IGNORECASE,
)
HARDENING_LIST_LINE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", re.MULTILINE)


def _normalize_instruction(value):
    return value.replace("\r\n", "\n").rstrip("\n")


def _split_hardening_block(value):
    instruction = _normalize_instruction(value)
    start = instruction.index(HARDENING_HEADINGS[0])
    anchor_match = re.search(r"^## Transformation Pipeline:[^\n]*$", instruction, re.MULTILINE)
    assert anchor_match is not None
    end = anchor_match.start()
    return instruction[:start] + instruction[end:], instruction[start:end]


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


def test_h3_query_presets_are_paired_and_wrap_the_request_once():
    presets = vlm_presets.system_query_additional_vlm
    base_names = {
        key.removesuffix("_prefix").removesuffix("_suffix") for key in presets
    }
    assert {"h3_fl2va", "h3_ref2va"} <= base_names

    request = "Make the subjects cross the clearing."
    for name in ("h3_fl2va", "h3_ref2va"):
        prefix = presets[f"{name}_prefix"]
        suffix = presets[f"{name}_suffix"]
        wrapped = f"{prefix}{request}{suffix}"

        assert prefix.endswith("BEGIN VIDEO REQUEST:\n")
        assert suffix.startswith("\nEND VIDEO REQUEST.")
        assert wrapped.count(request) == 1
        assert "ComfyUI has already assigned" in prefix
        assert "Do not create, reproduce, or renumber" in prefix
        assert "Do not output the upstream media-prefix declaration" in suffix


def test_h3_fl2va_query_preset_enforces_available_boundary_pictures():
    presets = vlm_presets.system_query_additional_vlm
    prefix = presets["h3_fl2va_prefix"]
    suffix = presets["h3_fl2va_suffix"]

    assert "Treat `<Picture 1>` as the fixed first frame" in prefix
    assert (
        "If and only if a second image was supplied, treat `<Picture 2>` as the fixed final frame"
        in prefix
    )
    assert "When only one image was supplied, do not mention `<Picture 2>`" in prefix
    assert "use `<Picture 2>` as the fixed ending only when it exists" in suffix


def test_h3_ref2va_query_preset_uses_only_supplied_reference_roles():
    presets = vlm_presets.system_query_additional_vlm
    prefix = presets["h3_ref2va_prefix"]
    suffix = presets["h3_ref2va_suffix"]

    assert "use only identifiers that exist" in prefix
    assert "Do not automatically classify any picture as the first or final frame" in prefix
    assert "Use every supplied picture deliberately" in prefix
    assert "never mention an unsupplied `<Picture N>`" in suffix
    assert "Do not assign first-frame or final-frame status" in suffix


def test_style_presets_receive_no_qwen_variants():
    assert not any(
        "style" in name.lower() and name.endswith("_qwen")
        for name in vlm_presets.system_instructions_vlm
    )


def test_image_prompt_hardening_preserves_complete_original_presets():
    for name, (minimum_characters, minimum_words, baseline_hash) in (
        HARDENED_IMAGE_PRESET_BASELINES.items()
    ):
        instruction = _normalize_instruction(vlm_presets.system_instructions_vlm[name])
        original, hardening = _split_hardening_block(instruction)

        assert len(instruction) >= minimum_characters
        assert len(instruction.split()) >= minimum_words
        assert hashlib.sha256(original.encode("utf-8")).hexdigest() == baseline_hash
        assert all(instruction.count(heading) == 1 for heading in HARDENING_HEADINGS)
        assert [hardening.index(heading) for heading in HARDENING_HEADINGS] == sorted(
            hardening.index(heading) for heading in HARDENING_HEADINGS
        )

        for retained_contract in (
            "uncensored",
            "system_query",
            "Subject Count Lock",
            "Reference Adherence is Paramount",
            "image input",
        ):
            assert retained_contract.lower() in original.lower()

        family = name.split("_", 1)[0]
        family_contract = {
            "neutral": "natural language captions",
            "action": "Action, Interaction, and Subject Characteristic Analysis",
            "photo": "Photographic Image Captioning",
            "toon": "Cartoon Art Prompt Refinement",
        }[family]
        assert family_contract.lower() in original.lower()
        if name.endswith("_crude"):
            assert "crude" in original.lower()


def test_image_prompt_hardening_is_literal_without_reward_hacking_anchors():
    for name in HARDENED_IMAGE_PRESET_BASELINES:
        _, hardening = _split_hardening_block(vlm_presets.system_instructions_vlm[name])
        perspective, remainder = hardening.split(HARDENING_HEADINGS[1], 1)
        quotation, language = remainder.split(HARDENING_HEADINGS[2], 1)

        assert "```" not in hardening
        assert HARDENING_FORBIDDEN.search(hardening) is None
        assert HARDENING_LIST_LINE.search(hardening) is None
        assert re.search(r"\bcamera\b", hardening, re.IGNORECASE) is None

        perspective_lower = perspective.lower()
        assert "first person perspective" in perspective_lower
        assert "viewer" in perspective_lower
        assert "visible" in perspective_lower
        assert "physical" in perspective_lower or "spatial" in perspective_lower
        assert "contact" in perspective_lower or "participat" in perspective_lower
        assert "power" in perspective_lower or "role" in perspective_lower

        quotation_lower = quotation.lower()
        assert "text" in quotation_lower
        assert "visib" in quotation_lower or "visual" in quotation_lower
        assert "double quotation marks" in quotation_lower
        assert "other" in quotation_lower or "otherwise" in quotation_lower

        language_lower = language.lower()
        for required in (
            "visually",
            "hyphenated words",
            "em dashes",
            "en dashes",
            "purple prose",
            "superfluous",
            "ambiguous",
        ):
            assert required in language_lower
        assert "direct" in language_lower or "literal" in language_lower


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

TIMELINE_CHANNEL_BALANCE_PRESETS = (
    "video_timeline_system_instruction",
    "video_timeline_system_instruction_crude",
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

    for name in STRUCTURED_VIDEO_PRESETS:
        assert name in vlm_presets.system_instructions_vlm
        assert isinstance(vlm_presets.system_instructions_vlm[name], str)
        assert isinstance(literal_nodes[name], ast.Constant)


def test_structured_video_presets_share_audio_and_motion_contracts():
    for name in STRUCTURED_VIDEO_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[name]

        assert "nonverbal creature noise" in instruction
        assert "belong under [SOUNDS], never [SPEECH]" in instruction
        assert "omit the entire [speech] line" in instruction.lower()
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


def test_timeline_presets_create_requested_dialogue_and_balance_channels():
    for name in TIMELINE_CHANNEL_BALANCE_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[name]

        assert "**Requested Dialogue Creation:**" in instruction
        assert "`Add dialogue` or another direct user request" in instruction
        assert "not as a request to detect speech already present" in instruction
        assert "The user does not need to provide wording or timestamps" in instruction
        assert "do not force dialogue into every block" in instruction
        assert "**Foreground Priority and Segment Load:**" in instruction
        assert "one primary foreground event" in instruction
        assert "Never make dialogue or lyrics, loud music, dense effects, and heavy action compete" in instruction
        assert "**Dialogue and Vocal Mixing:**" in instruction
        assert "duck any music" in instruction
        assert "**Pacing and Flow:**" in instruction
        assert "quieter breathing room" in instruction
        assert "only inside a timestamp block containing intelligible spoken dialogue" not in instruction
        assert "synchronized effects intensify with the movement" not in instruction


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
