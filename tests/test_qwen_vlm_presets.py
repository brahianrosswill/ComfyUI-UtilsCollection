import pathlib
import sys
import types


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_qwen_vlm_preset_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_qwen_vlm_preset_test import qwen_vlm_nodes, qwen_vlm_presets


EXPECTED_PRESETS = {
    "neutral_compact",
    "action_compact",
    "photo_compact",
    "toon_compact",
}


def test_compact_qwen_presets_are_isolated_and_directive_focused():
    presets = qwen_vlm_presets.qwen_system_instructions_vlm

    assert set(presets) == EXPECTED_PRESETS
    for instruction in presets.values():
        lowered = instruction.lower()
        assert 1_000 < len(instruction) < 5_000
        assert "return only" in lowered
        assert "do not argue" in lowered or "without disputing" in lowered
        assert "developed by google" not in lowered
        assert "e621" not in lowered
        assert "danbooru" not in lowered
        assert "example" not in lowered
        assert "e.g." not in lowered
        assert "i.e." not in lowered

    photo = presets["photo_compact"].lower()
    assert "regardless of whether the source image is anime" in photo
    assert "do not argue that photographic wording is inaccurate" in photo


def test_compact_qwen_basic_node_uses_only_compact_collection():
    schema = qwen_vlm_nodes.UC_QwenVLMSysInstrPresets.define_schema()
    preset_input = schema.inputs[0]

    assert schema.node_id == "UC_QwenVLMSysInstrPresets"
    assert schema.display_name == "Qwen VLM System Instruction Presets"
    assert preset_input.display_name == "qwen_vlm_system_instruction_preset"
    assert set(preset_input.options) == EXPECTED_PRESETS
    assert qwen_vlm_nodes.UC_QwenVLMSysInstrPresets.execute("photo_compact").args == (
        qwen_vlm_presets.qwen_system_instructions_vlm["photo_compact"],
    )


def test_compact_qwen_advanced_node_places_system_override_last():
    schema = qwen_vlm_nodes.UC_QwenVLMSysInstrAdvPresets.define_schema()
    result = qwen_vlm_nodes.UC_QwenVLMSysInstrAdvPresets.execute(
        "neutral_compact",
        "system directive",
        "user directive",
    ).args[0]

    assert schema.node_id == "UC_QwenVLMSysInstrAdvPresets"
    assert schema.display_name == "Qwen VLM System Instruction Advanced Presets"
    assert [value.id for value in schema.inputs] == ["preset", "system_query", "user_query"]
    assert schema.inputs[0].display_name == "qwen_vlm_system_instruction_advanced_preset"
    assert result.index("user directive") < result.index("system directive")
    assert result.endswith("Highest-priority system override:\nsystem directive")
