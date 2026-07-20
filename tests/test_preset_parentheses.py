import pathlib
import sys
import types

import pytest


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_preset_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_preset_test import preset_nodes


PRESET_NODES = (
    preset_nodes.UC_SystemMessagePresets,
    preset_nodes.UC_InstructPromptPresets,
    preset_nodes.UC_BonusPromptPresets,
    preset_nodes.UC_SystemMessageVideoPresets,
    preset_nodes.UC_InstructPromptVideoPresets,
    preset_nodes.UC_BonusPromptVideoPresets,
    preset_nodes.UC_LegacyPromptPresets,
    preset_nodes.UC_EditTargetPresets,
    preset_nodes.UC_EditOpPresets,
    preset_nodes.UC_CameraShotPresets,
)


@pytest.mark.parametrize("node", PRESET_NODES)
def test_unified_compatible_preset_nodes_expose_escape_parentheses(node):
    inputs = {value.id: value for value in node.define_schema().inputs}
    assert "escape_parentheses" in inputs
    assert inputs["escape_parentheses"].default is False


def test_preset_parenthesis_escaping_is_optional_and_idempotent():
    text = r"PBR (metal), already \(glass\)"
    assert preset_nodes.escape_prompt_parentheses(text) == text
    assert preset_nodes.escape_prompt_parentheses(text, True) == r"PBR \(metal\), already \(glass\)"


@pytest.mark.parametrize("node", [preset_nodes.UC_SystemMessagePresets, preset_nodes.UC_SystemMessageVideoPresets])
def test_style_3d_pbr_parentheses_are_escaped(node):
    plain = node.execute("STYLE_3D", False).args[0]
    escaped = node.execute("STYLE_3D", True).args[0]
    assert "(PBR)" in plain
    assert r"\(PBR\)" in escaped
    assert "(PBR)" not in escaped.replace(r"\(PBR\)", "")


def test_unified_presets_forwards_named_boolean_output():
    schema = preset_nodes.UC_UnifiedPresets.define_schema()
    assert [value.id for value in schema.inputs] == ["preset", "escape_parentheses"]
    result = preset_nodes.UC_UnifiedPresets.execute("STYLE_3D", True)
    assert result.args == ("STYLE_3D", True)
