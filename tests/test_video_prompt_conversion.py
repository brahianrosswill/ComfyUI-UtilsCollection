import pathlib
import sys
import types

import pytest


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_video_prompt_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_video_prompt_test import (
    helper_functions,
    preset_nodes,
    utils_nodes,
)


ROLE_GUIDANCE = {
    "general": helper_functions.VIDEO_PROMPT_GENERAL_GUIDANCE,
    "system": helper_functions.VIDEO_PROMPT_SYSTEM_GUIDANCE,
    "instruction": helper_functions.VIDEO_PROMPT_INSTRUCTION_GUIDANCE,
    "bonus": helper_functions.VIDEO_PROMPT_BONUS_GUIDANCE,
}


def test_video_prompt_roles_preserve_the_complete_source():
    source = (
        "Watercolor painting, cel illustration, film still, first frame, digital artwork.\n\n"
        "Keep (literal) punctuation, spacing, and café text unchanged."
    )

    assert helper_functions.VIDEO_PROMPT_ROLES == (
        "general",
        "system",
        "instruction",
        "bonus",
    )
    for role, guidance in ROLE_GUIDANCE.items():
        result = helper_functions.to_video_prompt(source, role=role)

        assert result == f"{source}\n\n{guidance}"
        assert result[: len(source)] == source
        assert result.count(source) == 1
        assert result.endswith(guidance)
        for other_role, other_guidance in ROLE_GUIDANCE.items():
            if other_role != role:
                assert not result.endswith(other_guidance)

        for legacy_replacement in (
            "animated sequence",
            "animation frame",
            "motion clip",
        ):
            assert legacy_replacement not in result


@pytest.mark.parametrize("role", helper_functions.VIDEO_PROMPT_ROLES)
@pytest.mark.parametrize("text", ("", " ", "\n\t"))
def test_video_prompt_roles_keep_empty_input_empty(role, text):
    assert helper_functions.to_video_prompt(text, role=role) == ""


def test_video_prompt_role_rejects_unknown_values():
    with pytest.raises(
        ValueError,
        match="Unsupported video prompt role: 'unknown'",
    ):
        helper_functions.to_video_prompt("source", role="unknown")


def test_static_constraints_remain_source_text_and_gain_motion_interpretation():
    source = "Keep the pose, position, framing, and composition unchanged."
    result = helper_functions.to_video_prompt(source)

    assert result.startswith(f"{source}\n\n")
    assert "defining the opening state and continuity" in result
    assert "unless the user explicitly requests a frozen shot" in result
    assert "does not prohibit subsequent motion" in result


def test_every_style_triplet_preserves_its_exact_source_preset():
    system_sources = preset_nodes.UC_SystemMessagePresets.get_presets()
    instruction_sources = preset_nodes.UC_InstructPromptPresets.get_presets()
    bonus_sources = preset_nodes.UC_BonusPromptPresets.get_presets()
    shared = set(system_sources) & set(instruction_sources) & set(bonus_sources)

    assert len(shared) == 82
    node_contracts = (
        (
            preset_nodes.UC_SystemMessageVideoPresets,
            system_sources,
            helper_functions.VIDEO_PROMPT_SYSTEM_GUIDANCE,
        ),
        (
            preset_nodes.UC_InstructPromptVideoPresets,
            instruction_sources,
            helper_functions.VIDEO_PROMPT_INSTRUCTION_GUIDANCE,
        ),
        (
            preset_nodes.UC_BonusPromptVideoPresets,
            bonus_sources,
            helper_functions.VIDEO_PROMPT_BONUS_GUIDANCE,
        ),
    )

    for preset_name in sorted(shared):
        for node, sources, guidance in node_contracts:
            source = sources[preset_name]
            result = node.execute(preset_name, False).args[0]

            assert result == f"{source}\n\n{guidance}"
            assert result[: len(source)] == source
            assert result.count(source) == 1


def test_every_non_style_system_message_is_preserved_verbatim():
    sources = preset_nodes.UC_SystemMessagePresets.get_presets()
    extra_system_presets = (
        "F2_SYSTEM_MESSAGE",
        "F2_SYSTEM_MESSAGE_UPSAMPLING_I2I",
        "F2_SYSTEM_MESSAGE_UPSAMPLING_T2I",
    )

    for preset_name in extra_system_presets:
        source = sources[preset_name]
        result = preset_nodes.UC_SystemMessageVideoPresets.execute(
            preset_name, False
        ).args[0]

        assert result == (
            f"{source}\n\n{helper_functions.VIDEO_PROMPT_SYSTEM_GUIDANCE}"
        )
        assert result[: len(source)] == source


def test_standalone_video_prompt_node_exposes_role_without_breaking_backend_default():
    schema = utils_nodes.UC_ImageToVideoPrompt.define_schema()
    inputs = {value.id: value for value in schema.inputs}

    assert [value.id for value in schema.inputs] == ["text", "prompt_role"]
    assert [value.id for value in schema.outputs] == ["text"]
    assert inputs["prompt_role"].options == [
        "general",
        "system",
        "instruction",
        "bonus",
    ]
    assert inputs["prompt_role"].default == "general"
    assert utils_nodes.UC_ImageToVideoPrompt.execute("source").args == (
        f"source\n\n{helper_functions.VIDEO_PROMPT_GENERAL_GUIDANCE}",
    )

    for role, guidance in ROLE_GUIDANCE.items():
        assert utils_nodes.UC_ImageToVideoPrompt.execute(
            "source", role
        ).args == (f"source\n\n{guidance}",)


def test_video_preset_parentheses_are_escaped_after_guidance_composition(monkeypatch):
    source = "Source (detail)"
    monkeypatch.setattr(
        preset_nodes.UC_BonusPromptVideoPresets,
        "get_presets",
        classmethod(lambda cls: {"TEST": source}),
    )

    plain = preset_nodes.UC_BonusPromptVideoPresets.execute("TEST", False).args[0]
    escaped = preset_nodes.UC_BonusPromptVideoPresets.execute("TEST", True).args[0]

    assert plain == f"{source}\n\n{helper_functions.VIDEO_PROMPT_BONUS_GUIDANCE}"
    assert escaped == plain.replace("(", r"\(").replace(")", r"\)")
