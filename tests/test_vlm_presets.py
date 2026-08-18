import ast
import hashlib
import pathlib
import re
import runpy
import sys
import types


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_vlm_preset_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_vlm_preset_test import (
    vlm_experimental_presets,
    vlm_legacy_presets,
    vlm_nodes,
    vlm_presets,
)


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
APPROVED_PERSPECTIVE_BLOCK = """## Perspective and Spatial Description

Determine the source image's viewpoint from the complete visible composition, and preserve it unless the user explicitly requests a change. State the most specific perspective description supported by the resulting composition. Use an established perspective term when it accurately describes that composition; when it does not fully express the geometry, describe the geometry directly without forcing a category. Ground the viewpoint in concrete spatial relationships consistent with the source image and request, without inventing a viewing location or precision those inputs do not establish. When the resulting composition establishes that the viewing position belongs to a scene participant, explicitly state first person perspective. State whose viewpoint it is only when the source image or request establishes that identity, and never assign first person perspective to an external viewpoint. Describe the complete resulting spatial arrangement, preserving every unchanged visible relationship and applying every requested change. State the framing, each relevant subject's orientation and pose, and the placement, relative scale, overlap, occlusion, and depth of all relevant subjects and objects. Include only depth relationships established by the source image or request, without filling a fixed layer template. Describe every visible or requested action and interaction concretely, stating what each involved subject or object does and all established directions and physical responses. When contact occurs, state which bodies or parts meet and where and how they meet. Never replace these relationships with vague interaction wording or treat contact alone as proof of an abstract participant role. Keep every claim grounded in visible source content or an explicit user request. Do not introduce terminology for physical image capture devices unless the device itself is visible in the image or explicitly requested by the user."""
REMOVED_PERSPECTIVE_PHRASES = (
    "exact established perspective term",
    "precise established perspective term",
    "exact physical coordinates",
    "exact vantage location",
    "foreground, midground, and background",
    "foreground, middle ground, and background",
    "dynamic power balances",
    "roles of dominance",
    "power or physical roles",
)
HARDENING_FORBIDDEN = re.compile(
    r"\be\.g\.\b|\bi\.e\.\b|\bexamples?\b|\bfor instance\b|"
    r"\bsuch as\b|\bincluding\b|\bsample outputs?\b|\bphrase menus?\b|"
    r"\bchoose from\b|\bpossible (?:terms|phrases|values)\b",
    re.IGNORECASE,
)
HARDENING_LIST_LINE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", re.MULTILINE)
GENDER_ALIAS_CLAUSES = {
    "Andromorph": (
        "Fetish or prompt aliases for this anatomy are cuntboy and pussyboy. "
        "Use trans man or transgender male only when the person's identity is "
        "established by the image or request; never infer transgender identity "
        "from anatomy alone."
    ),
    "Gynomorph": (
        "Fetish or prompt aliases for this anatomy are shemale, dickgirl, and ts. "
        "Use trans woman, transgender woman, or transgender female only when the "
        "person's identity is established by the image or request; never infer "
        "transgender identity from anatomy alone."
    ),
    "Herm": (
        "Fetish or prompt aliases for this anatomy are herm, hermaphrodite, female "
        "hermaphrodite, futanari, and futa. Use intersex or intersex female only "
        "when the person's identity or status is established by the image or "
        "request; never infer intersex identity or status from anatomy alone."
    ),
    "Maleherm": (
        "Fetish or prompt aliases for this anatomy are maleherm, male herm, and "
        "male hermaphrodite. Use intersex or intersex male only when the person's "
        "identity or status is established by the image or request; never infer "
        "intersex identity or status from anatomy alone."
    ),
}
LEGACY_SYSTEM_PRESET_BASELINES = {
    "neutral_system_instruction_legacy": (16_742, 2_368, 101, 0, "f0392ea612e34f0bee656f4e9bb89f59f9c6fc82539be2e2205443dfbede2e5d"),
    "action_system_instruction_legacy": (17_003, 2_390, 100, 0, "da80afcc3d11dc82ee51c0c313cb95106d77a6ac56c09f428211141f41e0999f"),
    "photo_system_instruction_legacy": (17_981, 2_579, 90, 0, "fd258a3196add5071a24087ae26c21102f04c46eb6c52af788d9babd9ff3d419"),
    "toon_system_instruction_legacy": (19_177, 2_763, 93, 0, "83a76127ef224f3bbac1606d0811a1b6e4d09a0490bbe355c3ba542bbf5cba0a"),
    "ideogram_4_json_instruction_legacy": (23_102, 3_250, 189, 0, "231c8eba8ba095789ac66925e2e3227770f712b8def25c12b328050a427bbc59"),
    "ideogram_4_json_instruction_short_legacy": (23_383, 3_382, 151, 0, "4e7d68fa89bb9d8c194e874ce7769612e8653fa44e7d2a5b3b5ce2673e700ca6"),
    "ideogram_4_json_instruction_style_legacy": (25_419, 3_590, 175, 0, "1152ef24619dffa1cd7b1739adcc1a3bf1e000d0959d0430243948a54e917348"),
    "ideogram_4_json_instruction_color_legacy": (25_915, 3_646, 177, 0, "95348d4a51a408a913bb6805f6752c9fe31af50bf46756564ae2cd52b923a547"),
    "cinematic_dumb_intelligent_legacy": (13_938, 1_852, 0, 154, "b05b49068d6c5bc1f8c20a307b4e181492be700852fbe7199a5aa2f084c53a61"),
    "video_basic_system_instruction_legacy": (14_428, 2_093, 74, 0, "1a6d406975d2dcd9d67767e6644928a119bde76f4848ce1d91fc4d69482f78ea"),
    "video_8sec_system_instruction_legacy": (13_723, 2_012, 97, 0, "ba69dc8e1daa850417d0d540a5ff9615b521837bd089cfc4b5a8f94b6355b55c"),
    "video_struct_system_instruction_legacy": (12_505, 1_811, 74, 0, "569eef0d2fd42c2786cf0019b7deea99ae13fb0c46dd5a805fef0061653c4c8d"),
    "video_8part_struct_system_instruction_legacy": (16_026, 2_348, 113, 0, "1c1174839c05208b179ab9eaf66d87402d21e2901aaaa4191ccfc05d1bbfa25b"),
}


def _normalize_instruction(value):
    return value.replace("\r\n", "\n").rstrip("\n")


def _split_hardening_block(value):
    instruction = _normalize_instruction(value)
    start = instruction.index(HARDENING_HEADINGS[0])
    anchor_match = re.search(r"^## Transformation Pipeline:[^\n]*$", instruction, re.MULTILINE)
    assert anchor_match is not None
    end = anchor_match.start()
    return instruction[:start] + instruction[end:], instruction[start:end]


def _strip_gender_alias_clauses(value):
    for clause in GENDER_ALIAS_CLAUSES.values():
        value = value.replace(f" {clause}", "")
    return value


def test_legacy_system_presets_match_d9f47de_exactly():
    assert not any(name.endswith("_legacy") for name in vlm_presets.system_instructions_vlm)
    actual_legacy_names = set(vlm_legacy_presets.legacy_system_instructions_vlm)
    assert actual_legacy_names == set(LEGACY_SYSTEM_PRESET_BASELINES)

    for name, (characters, words, crlf, bare_lf, digest) in (
        LEGACY_SYSTEM_PRESET_BASELINES.items()
    ):
        value = vlm_legacy_presets.legacy_system_instructions_vlm[name]
        assert len(value) == characters
        assert len(value.split()) == words
        assert value.count("\r\n") == crlf
        assert value.count("\n") - value.count("\r\n") == bare_lf
        assert hashlib.sha256(value.encode("utf-8")).hexdigest() == digest


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


def test_character_transfer_query_presets_keep_provider_reference_contracts_separate():
    presets = vlm_presets.system_query_additional_vlm
    variants = {
        "character_transfer_gemma": ("[`image 1`]", "[`image 2`]"),
        "character_transfer_qwen": ("<Picture 1>", "<Picture 2>"),
    }

    for name, (composition_reference, character_reference) in variants.items():
        prefix = presets[f"{name}_prefix"]
        suffix = presets[f"{name}_suffix"]

        assert prefix.endswith('\\{"current edit request": "')
        assert suffix == presets["image2image_suffix"]
        assert prefix.count(composition_reference) >= 5
        assert prefix.count(character_reference) >= 5
        assert "ordered, non-interchangeable sources" in prefix
        assert "permanently separate roles" in prefix
        assert "Never reverse, merge, weaken, or reinterpret these assignments" in prefix
        assert "Never identify, name, retain, or describe that subject's face" in prefix
        assert "Using your broad knowledge, identify that subject accurately" in prefix
        assert "intellectual property, theme, visual style, known media concept" in prefix
        assert "This is complete subject replacement" in prefix
        assert "No visual trait belonging to the subject" in prefix
        assert "Write the response as a direct description of the finished result" in prefix
        assert "who was always present in the resulting scene" in prefix

    assert "<Picture" not in presets["character_transfer_gemma_prefix"]
    assert "[`image" not in presets["character_transfer_qwen_prefix"]


def test_raw_query_presets_preserve_instructions_without_request_carriers():
    wrapped = vlm_presets.system_query_additional_vlm
    raw = vlm_presets.system_query_raw_vlm
    normalize = lambda value: value.replace("\r\n", "\n")
    expected_names = {
        key.removesuffix("_prefix").removesuffix("_suffix") for key in wrapped
    }
    assert set(raw) == expected_names

    json_carriers = {
        "character_transfer_gemma": "current edit request",
        "character_transfer_qwen": "current edit request",
        "ideogram_4": "current request",
        "image2image": "current edit request",
        "text2image": "current request",
        "video_basic": "current video request",
    }
    json_suffix = '\r\n\u2060 \u2060\u2060"\\}'
    for name, request_key in json_carriers.items():
        carrier = f' \\{{"{request_key}": "'
        assert wrapped[f"{name}_suffix"] == json_suffix
        assert raw[name] == wrapped[f"{name}_prefix"].removesuffix(carrier)

    for name in ("image2image_qwen", "text2image_qwen"):
        assert wrapped[f"{name}_suffix"] == ""
        assert normalize(raw[name]) == normalize(wrapped[f"{name}_prefix"]).removesuffix(
            " Current request:\n"
        )

    for name in (
        "h3_fl2va",
        "h3_ref2va",
        "h3_ref2va_alt",
        "h3_fl2va_experimental",
        "h3_ref2va_experimental",
    ):
        prefix = normalize(wrapped[f"{name}_prefix"]).removesuffix(
            "\n\nBEGIN VIDEO REQUEST:\n"
        )
        suffix = normalize(wrapped[f"{name}_suffix"]).removeprefix(
            "\nEND VIDEO REQUEST.\n\n"
        )
        assert normalize(raw[name]) == f"{prefix}\n\n{suffix}"

    t2va_prefix = wrapped["h3_t2va_prefix"].removesuffix(
        "\r\n\r\nBEGIN VIDEO REQUEST:\r\n"
    )
    t2va_suffix = wrapped["h3_t2va_suffix"].removeprefix(
        "\r\nEND VIDEO REQUEST.\r\n\r\n"
    )
    assert raw["h3_t2va"] == f"{t2va_prefix}\r\n\r\n{t2va_suffix}"

    forbidden = (
        '\\{"current request": "',
        '\\{"current edit request": "',
        '\\{"current video request": "',
        "Current request:",
        "BEGIN VIDEO REQUEST",
        "END VIDEO REQUEST",
    )
    assert all(
        value and not any(marker in value for marker in forbidden)
        for value in raw.values()
    )
    assert "<Picture" not in raw["character_transfer_gemma"]
    assert "[`image" not in raw["character_transfer_qwen"]


def test_raw_query_preset_node_outputs_selected_instruction_directly():
    schema = vlm_nodes.UC_VLMSysQueryRawPresets.define_schema()

    assert schema.node_id == "UC_VLMSysQueryRawPresets"
    assert schema.display_name == "VLM System Query Raw Presets"
    assert schema.category == "advanced/text"
    assert [value.id for value in schema.inputs] == ["preset"]
    assert schema.inputs[0].display_name == "vlm_system_query_raw_preset"
    assert schema.inputs[0].options == sorted(vlm_presets.system_query_raw_vlm)
    assert schema.outputs[0].display_name == "system_query"

    preset = schema.inputs[0].options[0]
    assert vlm_nodes.UC_VLMSysQueryRawPresets.execute(preset).args == (
        vlm_presets.system_query_raw_vlm[preset],
    )


def test_experimental_system_instruction_nodes_are_dedicated():
    presets = vlm_experimental_presets.system_instructions_vlm_experimental
    basic = vlm_nodes.UC_VLMSysInstrPresetsExperimental.define_schema()
    advanced = vlm_nodes.UC_VLMSysInstrAdvPresetsExperimental.define_schema()
    preset = sorted(presets)[0]

    assert basic.node_id == "UC_VLMSysInstrPresetsExperimental"
    assert basic.display_name == "VLM System Instruction Presets Experimental"
    assert basic.inputs[0].options == sorted(presets)
    assert advanced.node_id == "UC_VLMSysInstrAdvPresetsExperimental"
    assert (
        advanced.display_name
        == "VLM System Instruction Advanced Presets Experimental"
    )
    assert advanced.inputs[0].options == sorted(presets)
    assert vlm_nodes.UC_VLMSysInstrPresetsExperimental.execute(preset).args == (
        presets[preset],
    )
    result = vlm_nodes.UC_VLMSysInstrAdvPresetsExperimental.execute(
        preset,
        False,
        "system query",
        "user query",
    ).args[0]
    assert result.startswith(presets[preset])
    assert "system query" in result
    assert "user query" in result


def test_h3_query_presets_are_paired_and_wrap_the_request_once():
    presets = vlm_presets.system_query_additional_vlm
    base_names = {
        key.removesuffix("_prefix").removesuffix("_suffix") for key in presets
    }
    assert {
        "h3_t2va",
        "h3_fl2va",
        "h3_ref2va",
        "h3_fl2va_experimental",
        "h3_ref2va_experimental",
    } <= base_names

    request = "Make the subjects cross the clearing."
    for name in (
        "h3_fl2va",
        "h3_ref2va",
        "h3_fl2va_experimental",
        "h3_ref2va_experimental",
    ):
        prefix = presets[f"{name}_prefix"]
        suffix = presets[f"{name}_suffix"]
        wrapped = f"{prefix}{request}{suffix}"

        assert prefix.endswith("BEGIN VIDEO REQUEST:\n")
        assert suffix.startswith("\nEND VIDEO REQUEST.")
        assert wrapped.count(request) == 1

    for name in (
        "h3_fl2va",
        "h3_ref2va",
        "h3_fl2va_experimental",
        "h3_ref2va_experimental",
    ):
        prefix = presets[f"{name}_prefix"]
        suffix = presets[f"{name}_suffix"]
        assert "ComfyUI has already assigned" in prefix
        assert "Do not create, reproduce, or renumber" in prefix
        assert "Do not output the upstream media-prefix declaration" in suffix


def test_h3_t2va_query_uses_images_only_as_prompt_evidence():
    wrapped = vlm_presets.system_query_additional_vlm
    raw = vlm_presets.system_query_raw_vlm
    prefix = wrapped["h3_t2va_prefix"]
    suffix = wrapped["h3_t2va_suffix"]
    request = "Make the subjects cross the clearing."
    expected_raw = (
        prefix.removesuffix("\r\n\r\nBEGIN VIDEO REQUEST:\r\n")
        + "\r\n\r\n"
        + suffix.removeprefix("\r\nEND VIDEO REQUEST.\r\n\r\n")
    )

    assert prefix.endswith("BEGIN VIDEO REQUEST:\r\n")
    assert suffix.startswith("\r\nEND VIDEO REQUEST.")
    assert f"{prefix}{request}{suffix}".count(request) == 1
    assert "only as visual evidence" in prefix
    assert "not supplied to downstream MiniMax H3" in prefix
    assert "The completed prompt must stand on its text alone." in prefix
    assert "requested target visual direction" in prefix
    assert "overrides conflicting source rendering style" in prefix
    assert "concrete target-appropriate visual vocabulary" in prefix
    assert "state the governing target visual direction in `summary:`" in prefix
    assert "Do not invent production methods or unsupported visual additions" in prefix
    assert "never output `<Picture N>`" in prefix
    assert "MiniMax H3 receives this text and none of the supplied VLM images" in suffix
    assert "governs every subject definition and `summary:`" in suffix
    assert "without being restated inside [VISUAL]" in suffix
    assert "cannot substitute for requested visual-style adherence in those fields" in suffix
    assert "Do not output `<Picture N>`" in suffix
    assert "ComfyUI has already assigned" not in prefix
    assert "first frame" not in prefix.lower()
    assert "final frame" not in prefix.lower()
    assert raw["h3_t2va"] == expected_raw
    assert "BEGIN VIDEO REQUEST" not in raw["h3_t2va"]
    assert "END VIDEO REQUEST" not in raw["h3_t2va"]
    assert all(
        value.count("\n") == value.count("\r\n")
        for value in (prefix, suffix, raw["h3_t2va"])
    )
    for value in (prefix, suffix, raw["h3_t2va"]):
        lowered = value.lower()
        assert "example:" not in lowered
        assert "e.g." not in lowered
        assert "i.e." not in lowered
    assert "h3_t2va" in vlm_nodes.UC_VLMSysQueryAddPresets.get_presets()
    assert "h3_t2va" in vlm_nodes.UC_VLMSysQueryRawPresets.define_schema().inputs[0].options

    source = ast.parse(
        (CUSTOM_NODE_ROOT / "vlm_presets.py").read_text(encoding="utf-8")
    )
    dictionaries = {
        target.id: node.value
        for node in source.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id in {"system_query_additional_vlm", "system_query_raw_vlm"}
    }
    wrapped_literals = {
        ast.literal_eval(key): value
        for key, value in zip(
            dictionaries["system_query_additional_vlm"].keys,
            dictionaries["system_query_additional_vlm"].values,
        )
        if isinstance(key, ast.Constant)
    }
    raw_literals = {
        ast.literal_eval(key): value
        for key, value in zip(
            dictionaries["system_query_raw_vlm"].keys,
            dictionaries["system_query_raw_vlm"].values,
        )
        if isinstance(key, ast.Constant)
    }
    assert isinstance(wrapped_literals["h3_t2va_prefix"], ast.Constant)
    assert isinstance(wrapped_literals["h3_t2va_suffix"], ast.Constant)
    assert isinstance(raw_literals["h3_t2va"], ast.Constant)


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


def test_h3_ref2va_alt_preserves_explicit_timeline_and_replacement_contract():
    wrapped = vlm_presets.system_query_additional_vlm
    raw = vlm_presets.system_query_raw_vlm["h3_ref2va_alt"]
    prefix = wrapped["h3_ref2va_alt_prefix"]
    suffix = wrapped["h3_ref2va_alt_suffix"]

    for required in (
        "explicitly states `<Picture N> at TIMESTAMP`",
        "Preserve every explicit association",
        "without explicit timestamp associations as independent references",
        "references that precede all timeline samples",
        "Never infer this partition from input position",
        "source motion, pose progression, interaction",
        "Use independent `<Picture N>` references as provenance",
        "create one final subject",
        "identity and appearance come from the independent Picture",
        "Describe the final subject as continuously present",
        "Determine the governing visual style",
    ):
        assert required in prefix
    for required in (
        "Preserve every explicit Picture/timestamp association",
        "output exactly that segment count",
        "copy each supplied start literally",
        "Do not replace supplied starts with equal-duration divisions",
        "Depict the completed final subject continuously",
        "without media bookkeeping",
    ):
        assert required in suffix
    assert "<Video 1>" not in prefix + suffix + raw
    assert "numbered `<Picture 1>`" not in prefix + suffix + raw
    assert "every later image" not in prefix + suffix + raw
    assert "BEGIN VIDEO REQUEST:" not in raw
    assert "END VIDEO REQUEST." not in raw
    assert "BEGIN VIDEO REQUEST:" not in raw
    assert "END VIDEO REQUEST." not in raw


def test_h3_mixed_ref2va_uses_explicit_picture_timeline_associations():
    wrapped = vlm_presets.system_query_additional_vlm
    raw = vlm_presets.system_query_raw_vlm["h3_mixed_ref2va"]
    prefix = wrapped["h3_mixed_ref2va_prefix"]
    suffix = wrapped["h3_mixed_ref2va_suffix"]

    assert "h3_mixed_ref2va" in vlm_nodes.UC_VLMSysQueryAddPresets.get_presets()
    assert "h3_mixed_ref2va" in (
        vlm_nodes.UC_VLMSysQueryRawPresets.define_schema().inputs[0].options
    )
    assert "video_timeline_minimax_h3_mixed_system_instruction" in (
        vlm_presets.system_instructions_vlm
    )
    for required in (
        "exact `<Picture N>` identifier already assigned by ComfyUI",
        "explicit `<Picture N> at TIMESTAMP` associations",
        "exactly the Pictures named",
        "reference Pictures before the first timeline sample",
        "segment count only to validate",
        "Never infer the partition from input position",
        "assumed contiguous identifier range",
        "source motion, pose progression, interaction",
        "create one final subject",
    ):
        assert required in prefix
    for required in (
        "Output exactly the declared segment count",
        "Copy each supplied start timestamp literally",
        "Retain every existing `<Picture N>` identifier",
        "Never emit `<Video N>`",
        "Depict the final subject continuously",
        "final subject's identity, appearance, motion, scene, style, and continuity",
        "without media bookkeeping",
    ):
        assert required in suffix


def test_h3_mixed_system_instruction_preserves_media_partition_contract():
    instruction = vlm_presets.system_instructions_vlm[
        "video_timeline_minimax_h3_mixed_system_instruction"
    ]

    assert "video_timeline_minimax_h3_mixed_system_instruction" in (
        vlm_nodes.UC_VLMSysInstrPresets.get_presets()
    )
    for required in (
        "Existing Mixed Media",
        "regular user request",
        "leading ordered images",
        "<Video 1>",
        "<Picture N>",
        "Never write `<Video N>` inside a timestamp block",
        "subject_definitions:",
        "retention_analysis:",
    ):
        assert required in instruction


def test_readable_h3_reference_sources_match_runtime_presets():
    readable = runpy.run_path(str(CUSTOM_NODE_ROOT / "vlm_presets_vars.py"))

    def normalize(value):
        return value.replace("\r\n", "\n").replace("\r", "\n")

    assert normalize(
        vlm_presets.system_instructions_vlm[
            "video_timeline_minimax_h3_reference_system_instruction"
        ]
    ) == normalize(readable["VIDEO_TIMELINE_MINIMAX_H3_REFERENCE_SYSTEM_INSTRUCTION"])
    for name in ("H3_REF2VA", "H3_REF2VA_ALT", "H3_MIXED_REF2VA"):
        key = name.lower()
        prefix = normalize(readable[f"{name}_PREFIX"])
        suffix = normalize(readable[f"{name}_SUFFIX"])
        assert normalize(
            vlm_presets.system_query_additional_vlm[f"{key}_prefix"]
        ) == prefix
        assert normalize(
            vlm_presets.system_query_additional_vlm[f"{key}_suffix"]
        ) == suffix
        expected_raw = (
            prefix.removesuffix("\n\nBEGIN VIDEO REQUEST:\n")
            + "\n\n"
            + suffix.removeprefix("\nEND VIDEO REQUEST.\n\n")
        )
        assert normalize(vlm_presets.system_query_raw_vlm[key]) == expected_raw


def test_h3_reference_alt_assembled_context_matches_structured_picture_request():
    instruction = vlm_presets.system_instructions_vlm[
        "video_timeline_minimax_h3_reference_alt_system_instruction"
    ]
    prefix = vlm_presets.system_query_additional_vlm["h3_ref2va_alt_prefix"]
    suffix = vlm_presets.system_query_additional_vlm["h3_ref2va_alt_suffix"]
    request = (
        "Subject in <Picture 1> should replace subject in segment images. "
        "Target video duration is 12.309 seconds divided into 5 segments. "
        "Reference each image with <Picture 2> at 00.000s, <Picture 3> at "
        "03.066s, <Picture 4> at 06.131s, <Picture 5> at 09.197s and "
        "<Picture 6> at 12.262s."
    )
    assembled = instruction + prefix + request + suffix

    assert assembled.count(request) == 1
    assert instruction.count("{user_query}") == 8
    assert instruction.count("{system_query}") == 1
    assert "regular user request, not this marker, is authoritative" in instruction
    assert "exactly those starts and no additional section boundaries" in instruction
    assert "copy each start literally" in instruction
    assert "exact requested duration as the final end" in instruction
    assert "Preserve the request's timestamp precision" in instruction
    assert "sample identity is analysis-only and receives no subject alias" in instruction
    assert "final Picture-defined subject from the first frame through the last" in instruction
    assert "without emitting `<Picture N>` or `<Video N>` identifiers" in instruction
    assert "Do not replace supplied starts with equal-duration divisions" in suffix
    for competing in (
        "replacement subject",
        "displaced sample identity",
        "on-screen swap",
        "identity swap",
        "transformation, or reversion",
    ):
        assert competing not in instruction + prefix + suffix


def test_h3_ref2va_experimental_query_keeps_regression_snapshot():
    presets = vlm_presets.system_query_additional_vlm
    prefix = presets["h3_ref2va_experimental_prefix"]
    suffix = presets["h3_ref2va_experimental_suffix"]

    assert "Use `<Picture N>` as source provenance" in prefix
    assert "never as a timeline-segment anchor" in prefix
    assert "only in that subject or reference's first complete definition" in prefix
    assert "Do not repeat picture identifiers at each timeline interval" in suffix
    assert "wherever that reference materially controls" not in prefix


def test_h3_query_experimental_snapshots_are_static_and_complete():
    wrapped = vlm_presets.system_query_additional_vlm
    raw = vlm_presets.system_query_raw_vlm

    assert wrapped["h3_fl2va_experimental_prefix"] == wrapped["h3_fl2va_prefix"]
    assert wrapped["h3_fl2va_experimental_suffix"] == wrapped["h3_fl2va_suffix"]
    assert raw["h3_fl2va_experimental"] == raw["h3_fl2va"]
    assert wrapped["h3_ref2va_prefix"] != wrapped["h3_ref2va_experimental_prefix"]
    assert wrapped["h3_ref2va_suffix"] != wrapped["h3_ref2va_experimental_suffix"]
    assert raw["h3_ref2va"] != raw["h3_ref2va_experimental"]
    assert hashlib.sha256(
        wrapped["h3_ref2va_experimental_prefix"].encode()
    ).hexdigest() == "49069b640d3b721871205cd3fe1b74d962bcef42fb77bed5b257a92f88a1ef4a"
    assert hashlib.sha256(
        wrapped["h3_ref2va_experimental_suffix"].encode()
    ).hexdigest() == "cd232756a8c76dbb2b422c47d8fbcc0057081c2f7448e11a27f2e814a4827bab"
    assert hashlib.sha256(
        raw["h3_ref2va_experimental"].encode()
    ).hexdigest() == "55b59d353d97e3ca9301745e82e36c255e7dda59bbdd861b5287d9408a1a1434"


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
        baseline_original = _strip_gender_alias_clauses(original)

        assert len(instruction) >= minimum_characters
        assert len(instruction.split()) >= minimum_words
        assert hashlib.sha256(baseline_original.encode("utf-8")).hexdigest() == baseline_hash
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


def test_action_perspective_hardening_matches_neutral_baseline():
    perspectives = {}
    for name in ("neutral_system_instruction", "action_system_instruction"):
        _, hardening = _split_hardening_block(
            vlm_presets.system_instructions_vlm[name]
        )
        perspectives[name] = hardening.split(HARDENING_HEADINGS[1], 1)[0]

    assert perspectives["action_system_instruction"] == perspectives[
        "neutral_system_instruction"
    ]


def test_gender_taxonomy_presets_include_qualified_equivalent_terms():
    taxonomy_presets = {
        name: value
        for name, value in vlm_presets.system_instructions_vlm.items()
        if "Gynomorph" in value
    }

    assert len(taxonomy_presets) == 23
    for instruction in taxonomy_presets.values():
        for label, clause in GENDER_ALIAS_CLAUSES.items():
            matching_lines = [line for line in instruction.splitlines() if label in line]
            assert len(matching_lines) == 1
            assert clause in matching_lines[0]


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
        assert perspective.rstrip() == APPROVED_PERSPECTIVE_BLOCK
        for removed_phrase in REMOVED_PERSPECTIVE_PHRASES:
            assert removed_phrase not in perspective_lower

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
    "video_timeline_minimax_h3_reference_alt_system_instruction",
)

H3_CAMERA_CONTINUITY_PRESETS = (
    "video_timeline_minimax_h3_base_system_instruction",
    "video_timeline_minimax_h3_t2va_system_instruction",
    "video_timeline_minimax_h3_reference_system_instruction",
)

TIMELINE_CHANNEL_BALANCE_PRESETS = (
    "video_timeline_system_instruction",
    "video_timeline_system_instruction_crude",
    "video_timeline_minimax_h3_base_system_instruction",
)

EXPERIMENTAL_VIDEO_PRESET_HASHES = {
    "video_timeline_system_instruction": (
        "cc3f2d9f32894a190a1509f9406c30320bb81250b93af7f696207f6e17d58ca8"
    ),
    "video_timeline_system_instruction_crude": (
        "3bc602b0a6ef184e27002d3b6250a3b24a708dad92424a4159729881842f1a6f"
    ),
    "video_timeline_minimax_h3_base_system_instruction": (
        "951501e17444450e5946e4bf2f2f9e7998cbdf0724a9fb5e6ff08e49f7e63eef"
    ),
    "video_timeline_minimax_h3_reference_system_instruction": (
        "05cc45e16bb58a81d1e2dab073f9af507640ecd6238d776c7e1ff73f15403917"
    ),
}

STABLE_VIDEO_PRESET_HASHES = {
    "video_timeline_system_instruction": (
        "57b6a726a9fce905f3003f67311cac02a28eb2c9d21ae72f66e8442a0f99e632"
    ),
    "video_timeline_system_instruction_crude": (
        "7469a2e17d36225c8b948b36bf47cdfb432c507d89efcb44a153d63b50120f0a"
    ),
    "video_timeline_minimax_h3_base_system_instruction": (
        "fadfdce95e9c24e7d473f7cf435d3e6d55a9063e9288034cef93ad5658cfdd95"
    ),
    "video_timeline_minimax_h3_reference_system_instruction": (
        "aa0faedbd0bc04095e8702ff6fec48717111f0e985d3be9667fac84dfadb5c5f"
    ),
    "video_timeline_minimax_h3_reference_alt_system_instruction": (
        "3ea27fee780bd00a557b1c2d7f1c6d2917e4d94d706484c2434112f7a2634605"
    ),
    "video_timeline_minimax_h3_mixed_system_instruction": (
        "5a97315c659999331ad1f9ae85d96f37c8d6729f433d7da2ceb2037dd6c0c098"
    ),
}


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


def test_experimental_video_presets_are_independent_literal_snapshots():
    source = ast.parse(
        (CUSTOM_NODE_ROOT / "vlm_experimental_presets.py").read_text(
            encoding="utf-8"
        )
    )
    preset_dict = next(
        node.value
        for node in source.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "system_instructions_vlm_experimental"
            for target in node.targets
        )
    )
    literal_nodes = {
        ast.literal_eval(key): value
        for key, value in zip(preset_dict.keys, preset_dict.values)
        if isinstance(key, ast.Constant)
    }

    assert set(literal_nodes) == set(EXPERIMENTAL_VIDEO_PRESET_HASHES)
    assert all(isinstance(value, ast.Constant) for value in literal_nodes.values())


def test_stable_and_experimental_video_preset_hashes_are_locked():
    stable = vlm_presets.system_instructions_vlm
    experimental = vlm_experimental_presets.system_instructions_vlm_experimental

    assert set(experimental) == set(EXPERIMENTAL_VIDEO_PRESET_HASHES)
    for name, expected_hash in STABLE_VIDEO_PRESET_HASHES.items():
        assert hashlib.sha256(stable[name].encode()).hexdigest() == expected_hash
    for name, expected_hash in EXPERIMENTAL_VIDEO_PRESET_HASHES.items():
        assert hashlib.sha256(experimental[name].encode()).hexdigest() == expected_hash


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
    assert "[00.00s-01.00s]:" in instruction
    assert "[01.00s-02.50s]:" in instruction
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
        assert "use no fixed number of sections" in instruction.lower()
        assert "Every range touches the next without a gap or overlap" in instruction
        assert "final range ends at the exact total duration" in instruction
        if name == "video_timeline_minimax_h3_reference_alt_system_instruction":
            assert "[START-END]:" in instruction
            assert "copy each start literally" in instruction
        else:
            assert "[00.00s-00.00s]:" in instruction
        assert "[SPEECH]:" in instruction
        assert "<d>[Language]" in instruction


def test_minimax_h3_timeline_presets_mark_actual_cuts_with_shot_references():
    for name in H3_CAMERA_CONTINUITY_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[name]

        assert "**Shot Continuity:**" in instruction
        assert (
            "Introduce sequential `[Shot N]` markers inside [VISUAL] only when the "
            "scene actually cuts or transitions."
        ) in instruction
        assert "The timestamp range remains the authoritative timing structure." in instruction


def test_minimax_h3_timeline_presets_require_segment_music_contract():
    for name in H3_CAMERA_CONTINUITY_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[name]

        assert "[VISUAL], optional [SPEECH], [SOUNDS], and optional [MUSIC]" in instruction
        assert "Omit [MUSIC] from segments with no music specific to them." in instruction
        assert "When music is specific to a timestamp block, write [MUSIC] after [SOUNDS]." in instruction
        assert "State the type of music for that segment." in instruction
        assert (
            "Mention <Subject N> in [MUSIC] only when that actual subject is "
            "playing the music; otherwise state only the music type."
        ) in instruction
        assert "as the whole-video summary of music specified in the timeline." in instruction
        assert "Do not introduce music absent from the timeline." in instruction


def test_stable_timeline_presets_require_zero_padded_two_decimal_seconds():
    for name in TIMELINE_CHANNEL_BALANCE_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[name]

        assert "first range begins at `00.00s`" in instruction
        assert "[00.00s-00.00s]:" in instruction
        assert "total elapsed seconds" in instruction
        assert "at least two integer digits" in instruction
        assert "exactly two decimal digits" in instruction
        assert "zero-padded two-decimal" in instruction
        assert "fewest integer digits needed" not in instruction
        assert "[0s-" not in instruction
        assert "[start-end]:" not in instruction


def test_experimental_timeline_presets_keep_minimal_width_snapshot():
    for instruction in (
        vlm_experimental_presets.system_instructions_vlm_experimental.values()
    ):
        assert "first range begins at `0.00s`" in instruction
        assert "[0.00s-0.00s]:" in instruction
        assert "fewest integer digits needed" in instruction
        assert "zero-padded two-decimal" not in instruction


def test_minimax_h3_t2va_uses_general_standalone_timeline_contract():
    name = "video_timeline_minimax_h3_t2va_system_instruction"
    instruction = vlm_presets.system_instructions_vlm[name]
    reference_instruction = vlm_presets.system_instructions_vlm[
        "video_timeline_minimax_h3_reference_system_instruction"
    ]
    base_instruction = vlm_presets.system_instructions_vlm[
        "video_timeline_minimax_h3_base_system_instruction"
    ]
    prefix = vlm_presets.system_query_additional_vlm["h3_t2va_prefix"]
    suffix = vlm_presets.system_query_additional_vlm["h3_t2va_suffix"]
    raw = vlm_presets.system_query_raw_vlm["h3_t2va"]
    basic_schema = vlm_nodes.UC_VLMSysInstrPresets.define_schema()
    advanced_schema = vlm_nodes.UC_VLMSysInstrAdvPresets.define_schema()

    assert name in basic_schema.inputs[0].options
    assert name in advanced_schema.inputs[0].options
    assert instruction != reference_instruction
    assert instruction != base_instruction

    assert "MiniMax H3 Standalone Text-to-Video" in instruction
    assert "MiniMax H3 receives only the completed text" in instruction
    assert "receives none of these images" in instruction
    assert "VLM-Only Visual Evidence" in instruction
    assert "Standalone Prompt Boundary" in instruction
    assert "Never emit `<Picture N>`" in instruction
    assert instruction.count("<Picture N>") == 1
    assert "Complete First-Use Definitions" in instruction
    assert "**Requested Target Visual Style:**" in instruction
    assert "target direction governs the completed video" in instruction
    assert "overrides conflicting source-image rendering style" in instruction
    assert "concrete visual language appropriate to the requested target style" in instruction
    assert "Do not invent production methods or unsupported visual additions" in instruction
    assert "State the governing target visual style" in instruction
    assert "Do not restate the global target style inside [VISUAL]" in instruction
    assert "Concrete lighting or color changes may appear there only when materially relevant" in instruction
    assert "cannot substitute for target-style information in the subject definitions and summary" in instruction
    assert "preserve supported source-image style evidence" in instruction
    assert "requested target-style adherence in every subject definition and in `summary:`" in instruction
    assert "no global target-style restatement inside [VISUAL]" in instruction
    assert "opening [VISUAL] block" not in instruction
    assert "maintain it through every later [VISUAL] block" not in instruction
    assert "visual style, and physical state" not in instruction
    assert "Keep [VISUAL] focused on scene state, action, interaction, camera movement" in instruction
    assert "Write `summary:` immediately after the completed timeline" in instruction
    assert "governing target visual style, medium, era, and subject presentation" in instruction
    assert (
        "subject_definitions:\r\n<Subject 1>: complete definition\r\n"
        "<Subject 2>: complete definition"
    ) in instruction
    assert "beginning in column one" in instruction
    assert "Do not place a bullet, numbering prefix, indentation" in instruction
    assert "`<Subject N>`" not in instruction
    assert "immutable semantic reference token, never as a word or name" in instruction
    assert "Never place an apostrophe, possessive marker" in instruction
    assert "Correct possession form: the red sash worn by <Subject 1>." in instruction
    assert "Forbidden possession form: <Subject 1>'s red sash." in instruction
    assert "place surrounding grammar outside it" not in instruction
    assert instruction.count("{user_query}") == 5
    assert instruction.count("{system_query}") == 1
    lowered_instruction = instruction.lower()
    assert "example:" not in lowered_instruction
    assert "e.g." not in lowered_instruction
    assert "i.e." not in lowered_instruction

    fields = (
        "subject_definitions:",
        "detailed_description:",
        "summary:",
        "overall_soundscape:",
        "non_diegetic_music:",
    )
    positions = [instruction.index(field) for field in fields]
    assert positions == sorted(positions)
    assert "exactly five top-level fields" in instruction
    assert "Place `Timeline:` immediately beneath `detailed_description:`" in instruction
    assert "Place `summary:` immediately after the complete timeline" in instruction
    assert "Do not enumerate, sequence, condense, restate, paraphrase" in instruction
    assert "duplicate timeline progression in `summary:`" in instruction

    assert "[0.00s-0.00s]:" in instruction
    assert "first range begins at `0.00s`" in instruction
    assert "Use the fewest integer digits needed" in instruction
    assert "exactly two decimal digits" in instruction
    for forbidden in (
        "[00.00s-00.00s]:",
        "MM:SS.mmm",
        "00:00.000",
        "ComfyUI constructs",
        "Existing Media References",
        "Explicit Picture Timestamp Mapping",
        "retention_analysis:",
        "integrated_multimodal_description:",
        "fully_preserved",
        "<Video N>",
        "<Audio N>",
    ):
        assert forbidden not in instruction

    assert "Use every ordered image supplied with this request" in prefix
    assert "BEGIN VIDEO REQUEST:" in prefix
    assert "END VIDEO REQUEST." in suffix
    assert "none of the supplied VLM images" in suffix
    assert "BEGIN VIDEO REQUEST:" not in raw
    assert "END VIDEO REQUEST." not in raw
    for value in (
        instruction,
        prefix,
        suffix,
        raw,
    ):
        assert value.count("\n") == value.count("\r\n")

    assert "retention_analysis:" in reference_instruction
    assert "integrated_multimodal_description:" in base_instruction

    source = ast.parse(
        (CUSTOM_NODE_ROOT / "vlm_presets.py").read_text(encoding="utf-8")
    )
    preset_dict = next(
        node.value
        for node in source.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "system_instructions_vlm"
            for target in node.targets
        )
    )
    literal_nodes = {
        ast.literal_eval(key): value
        for key, value in zip(preset_dict.keys, preset_dict.values)
        if isinstance(key, ast.Constant)
    }
    assert isinstance(literal_nodes[name], ast.Constant)


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
        "video_timeline_minimax_h3_reference_alt_system_instruction"
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
    assert "assign a media number" in instruction
    assert "renumber an existing media identifier" in instruction
    assert "Create and number <Subject N> aliases only" in instruction
    assert "Assign `<Picture N>`" not in instruction
    assert "Number each category independently" not in instruction
    assert "<Subject N>" in instruction
    assert "<Picture N>" in instruction
    assert "<Video N>" in instruction
    assert "<Audio N>" in instruction
    assert "A label never replaces the full subject" in instruction
    assert "**Governing Reference Style:**" in instruction
    assert "explicit requested style takes priority only where it conflicts" in instruction
    assert "When no conflict exists, preserve and state the supported source rendering style" in instruction
    assert "rendering medium as style evidence rather than immutable subject identity" in instruction
    assert "MiniMax H3 receives the referenced images" in instruction
    assert "visual vocabulary appropriate to the governing style" in instruction
    assert "Retain accurate source rendering-medium descriptions when that style remains active" in instruction
    assert "do not carry them into a conflicting requested target style" in instruction
    assert "Do not invent production methods or unsupported additions" in instruction
    assert "Place `Timeline:` immediately beneath `detailed_description:`" in instruction
    assert "In `summary:`, state the intended target video" in instruction
    assert "established reference relationships" in instruction
    assert "governing visual style, medium, era, and subject presentation" in instruction
    assert "write one concise paragraph using established <Subject N> aliases" in instruction
    assert "final subjects' identity, appearance, motion, scene, style, and continuity" in instruction
    assert "without emitting `<Picture N>` or `<Video N>` identifiers" in instruction
    assert "source identity that is absent from the completed target" in instruction
    assert "Do not include action choreography, event progression, transformation timing" in instruction
    assert "Do not repeat subject definitions, summary content, or exhaustive source description" in instruction
    assert "Do not invent production methods, construction details" in instruction
    assert "without restating the global governing style" in instruction
    assert "conditional source-style retention" in instruction
    assert "requested target-style precedence only where conflicts exist" in instruction
    assert "governing style in `summary:`" in instruction
    assert "concise media roles and continuity constraints only in `retention_analysis:`" in instruction
    assert "Keep action, transformation, event order, and timing exclusively inside" in instruction
    assert "no choreography, progression, timing, production methods" in instruction
    assert "correct use of downstream reference-image availability" in instruction
    assert "Do not invent task classifications or asset roles" in instruction
    assert "use the literal alias only at the subject's first introduction" in instruction
    assert "Otherwise use the subject's concise ordinary name" in instruction
    assert "Never repeat one alias multiple times in a timestamp block" in instruction
    assert "neither visibly supported nor explicitly introduced" in instruction
    assert "do not use repeated aliases as continuity reinforcement" in instruction
    assert "Preserve each subject alias throughout the output" not in instruction
    assert "wherever its role materially affects the current interval" not in instruction
    assert "**Atomic Subject Labels:**" in instruction
    assert (
        "subject_definitions:\r\n<Subject 1>: complete definition\r\n"
        "<Subject 2>: complete definition"
    ) in instruction
    assert "beginning in column one" in instruction
    assert "Do not place a bullet, numbering prefix, indentation" in instruction
    assert "`<Subject N>`" not in instruction
    assert "immutable semantic reference token, never as a word or name" in instruction
    assert "Never place an apostrophe, possessive marker" in instruction
    assert "Correct possession form: the red sash worn by <Subject 1>." in instruction
    assert "Forbidden possession form: <Subject 1>'s red sash." in instruction
    assert "\r\n" in instruction
    assert not re.search(r"(?<!\r)\n", instruction)
    assert instruction.count("{user_query}") == 8
    assert instruction.count("{system_query}") == 1
    lowered_instruction = instruction.lower()
    assert "example:" not in lowered_instruction
    assert "e.g." not in lowered_instruction
    assert "i.e." not in lowered_instruction
    source = (CUSTOM_NODE_ROOT / "vlm_presets.py").read_text(encoding="utf-8")
    assert re.search(
        r'^    "video_timeline_minimax_h3_reference_system_instruction": "',
        source,
        re.MULTILINE,
    )
    assert "Explicit Picture Timestamp Mapping" not in instruction
    assert "Place `summary:` immediately after the complete timeline" not in instruction
    assert "Do not write `<Picture N>` anywhere inside" not in instruction
    assert "never cite `<Picture N>` in a timestamp block" not in instruction
    assert "picture-to-timestamp or picture-to-shot declaration" not in instruction
    assert "separately assembled downstream H3 media prefix" not in instruction
    assert "fixed ordinal pairing" not in instruction
    assert "shot mapping was supplied" not in instruction
    assert "Sample Video Frames" not in instruction


def test_experimental_h3_reference_keeps_regression_contract():
    instruction = vlm_experimental_presets.system_instructions_vlm_experimental[
        "video_timeline_minimax_h3_reference_system_instruction"
    ]
    fields = (
        "subject_definitions:",
        "retention_analysis:",
        "detailed_description:",
        "summary:",
        "overall_soundscape:",
        "non_diegetic_music:",
    )

    assert [instruction.index(field) for field in fields] == sorted(
        instruction.index(field) for field in fields
    )
    assert "Place `summary:` immediately after the complete timeline" in instruction
    assert "as an internal picture-to-time map" in instruction
    assert "Do not write `<Picture N>` anywhere inside" in instruction
    assert "never cite `<Picture N>` in a timestamp block" in instruction
    assert "cite ComfyUI's existing `<Picture N>` identifiers only as subject provenance" in instruction
    assert "**Atomic Subject Labels:**" in instruction
    assert "Never append possessive markers" in instruction
    assert "without modifying the alias" in instruction


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
