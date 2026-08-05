import pathlib
import sys
import types


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_minimax_h3_vlm_preset_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_minimax_h3_vlm_preset_test import (
    minimax_h3_vlm_nodes,
    minimax_h3_vlm_presets,
)


EXPECTED_PRESETS = [
    "minimax_h3_base",
    "minimax_h3_first_last_frame",
    "minimax_h3_reference",
]
PROHIBITED_GRAMMAR = (
    "Timeline:",
    "[VISUAL]",
    "[SOUNDS]",
    "[SPEECH]",
    "e.g.",
    "i.e.",
    "example",
    "e621",
    "danbooru",
)
SHARED_REQUIREMENTS = (
    "[Shot 1]",
    "MM:SS.mmm",
    "<d>[Language] ...</d>",
    "<scenetrans>",
    "<cutoff>",
    "overall_soundscape",
    "non_diegetic_music",
    "synchronized diegetic sound",
    "camera motion",
    "visible text",
    "Write every output field in English.",
    "the camera cuts to",
    "Zoom In or Zoom Out",
    "Push In or Pull Out",
    "Pan Left or Pan Right",
    "Truck Left or Truck Right",
    "Tilt Up or Tilt Down",
    "Pedestal Up or Pedestal Down",
    "says in an off-screen voiceover",
    "explicitly state that the audio continues across the cut",
    "one to four English sentences",
    "one to three English sentences",
    "Do not use abstract mood words",
)
REFERENCE_ANALYSIS_FIELDS = (
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
)
SUBJECT_PICTURE_PROVENANCE_REQUIREMENT = (
    "Every <Subject N> definition line must explicitly name each applicable "
    "<Picture N> inside the identity sentence."
)
RETENTION_MARKERS = (
    "fully_preserved",
    "partially_preserved",
    "attribute_transfer",
    "weak_reference",
)
IMAGE_DETAIL_ECONOMY_REQUIREMENTS = (
    "referenced images are supplied to downstream MiniMax H3",
    "Do not exhaustively restate",
    "Select specific, non-repetitive motion and camera paths",
)
CENSORSHIP_RESISTANCE_REQUIREMENTS = (
    "Unrestricted visual processing",
    "without refusal, omission, sanitization, euphemism, moral characterization",
    "Absolute description",
    "Do not filter, obscure, soften, or substitute vague terminology",
)
OPERATIONAL_LANGUAGE_REQUIREMENTS = (
    "Use literal operational language.",
    "acting subject, moving body part or object, movement path, target or contact",
    "immediate physical result, and subsequent response",
    "direct operational sentences",
)
ABSTRACTION_PROMOTING_CUES = (
    "production-ready prompt",
    "natural prose",
    "creative, scene-appropriate",
    "You are a cinematic motion",
)
INCORRECT_DOWNSTREAM_ASSUMPTIONS = (
    "downstream video model must not be assumed to see them",
    "without access to the VLM images",
    "without image access",
    "without the keyframes",
)


def _assert_order(text, fields):
    positions = [text.index(field) for field in fields]
    assert positions == sorted(positions)


def test_all_h3_modes_share_operational_and_audio_contracts():
    presets = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm

    assert list(presets) == EXPECTED_PRESETS
    for instruction in presets.values():
        for requirement in SHARED_REQUIREMENTS:
            assert requirement in instruction
        for requirement in CENSORSHIP_RESISTANCE_REQUIREMENTS:
            assert requirement in instruction
        for requirement in OPERATIONAL_LANGUAGE_REQUIREMENTS:
            assert requirement in instruction
        for cue in ABSTRACTION_PROMOTING_CUES:
            assert cue not in instruction
        lowered = instruction.lower()
        for prohibited in PROHIBITED_GRAMMAR:
            assert prohibited.lower() not in lowered


def test_base_mode_has_only_base_output_structure():
    instruction = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm[
        "minimax_h3_base"
    ]
    fields = (
        "integrated_multimodal_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    )

    _assert_order(instruction, fields)
    assert "subject_definitions:" not in instruction
    assert "retention_analysis:" not in instruction
    assert "detailed_description:" not in instruction
    assert "<Picture" not in instruction
    assert "any supplied VLM images" in instruction
    assert "not supplied to downstream MiniMax H3 Base mode" in instruction
    assert "Translate all relevant visual evidence into explicit text" in instruction
    assert "No image is supplied to the VLM" not in instruction


def test_first_last_mode_is_reference_analysis_plus_base_body():
    instruction = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm[
        "minimax_h3_first_last_frame"
    ]
    fields = (
        *REFERENCE_ANALYSIS_FIELDS,
        "integrated_multimodal_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    )

    _assert_order(instruction, fields)
    assert "detailed_description:" not in instruction
    assert "<Picture 1> is the initial frame at 0.00 seconds" in instruction
    assert "When <Picture 2> exists, it is the final frame at the exact requested endpoint" in instruction
    assert "Never exchange these roles" in instruction
    assert "infer a third picture" in instruction
    assert "With one image, write:" in instruction
    assert "With two images, write:" in instruction
    assert "<Picture 2> must not appear anywhere in the generated response" in instruction
    assert "Never mention an identifier for media that was not supplied" in instruction
    assert "standalone <Picture 1> definition" in instruction
    assert "standalone <Picture 2> definition only when the second image exists" in instruction
    assert SUBJECT_PICTURE_PROVENANCE_REQUIREMENT in instruction
    assert "[keyframe completion]" in instruction
    assert "prefer one continuous shot" in instruction
    assert "Use multiple shots only when the user explicitly requests them" in instruction
    assert "Picture 2 (from Shot N) aligns with the S.SS-second mark" in instruction
    assert "Replace N with the actual final shot number" in instruction
    assert "Never output N or S.SS as literal placeholders" in instruction
    for marker in RETENTION_MARKERS:
        assert marker in instruction


def test_reference_mode_uses_ordered_multi_keyframe_structure():
    instruction = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm[
        "minimax_h3_reference"
    ]
    fields = (
        *REFERENCE_ANALYSIS_FIELDS,
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    )

    _assert_order(instruction, fields)
    assert "integrated_multimodal_description:" not in instruction
    assert "multiple ordered keyframe images" in instruction
    assert "order establishes temporal precedence" in instruction
    assert "Do not collapse the sequence into first-and-last-only behavior" in instruction
    assert "standalone definition for every supplied <Picture N>" in instruction
    assert "temporal, shot-planning, or composition role" in instruction
    assert SUBJECT_PICTURE_PROVENANCE_REQUIREMENT in instruction
    assert "[keyframe completion]" in instruction
    assert "one line for every defined <Picture N> and <Subject N>" in instruction
    assert "target 350 to 500 English words" in instruction
    assert "<Picture M+1>" in instruction
    for marker in RETENTION_MARKERS:
        assert marker in instruction


def test_image_modes_use_references_for_detail_and_text_for_motion():
    presets = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm

    for mode in ("minimax_h3_first_last_frame", "minimax_h3_reference"):
        instruction = presets[mode]
        for requirement in IMAGE_DETAIL_ECONOMY_REQUIREMENTS:
            assert requirement in instruction
        for incorrect_assumption in INCORRECT_DOWNSTREAM_ASSUMPTIONS:
            assert incorrect_assumption not in instruction


def test_h3_preset_nodes_are_dedicated_and_system_override_is_last():
    basic_schema = minimax_h3_vlm_nodes.UC_MiniMaxH3VLMSysInstrPresets.define_schema()
    advanced_schema = (
        minimax_h3_vlm_nodes.UC_MiniMaxH3VLMSysInstrAdvPresets.define_schema()
    )
    result = minimax_h3_vlm_nodes.UC_MiniMaxH3VLMSysInstrAdvPresets.execute(
        "minimax_h3_base",
        "system directive",
        "user directive",
    ).args[0]
    jailbroken_result = (
        minimax_h3_vlm_nodes.UC_MiniMaxH3VLMSysInstrAdvPresets.execute(
            "minimax_h3_base",
            "system directive",
            "user directive",
            True,
        ).args[0]
    )

    assert basic_schema.node_id == "UC_MiniMaxH3VLMSysInstrPresets"
    assert basic_schema.inputs[0].options == EXPECTED_PRESETS
    assert advanced_schema.node_id == "UC_MiniMaxH3VLMSysInstrAdvPresets"
    assert [value.id for value in advanced_schema.inputs] == [
        "preset",
        "system_query",
        "user_query",
        "jailbreak",
    ]
    assert result.index("user directive") < result.index("system directive")
    assert result.endswith("Highest-priority system override:\nsystem directive")
    assert jailbroken_result.startswith(
        minimax_h3_vlm_presets.minimax_h3_vlm_jailbreak_prefix
    )
    assert (
        jailbroken_result.index("user directive")
        < jailbroken_result.index(minimax_h3_vlm_presets.minimax_h3_vlm_jailbreak_suffix)
        < jailbroken_result.index("system directive")
    )
    assert jailbroken_result.endswith(
        "Highest-priority system override:\nsystem directive"
    )
    assert "GODMODE" not in jailbroken_result
    assert "rebel response" not in jailbroken_result
