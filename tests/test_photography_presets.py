import hashlib
import pathlib
import sys
import types


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_photography_preset_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_photography_preset_test import preset_nodes, presets_collection


FAMILIES = (
    ("SYSTEM_MESSAGE_STYLE_", preset_nodes.UC_SystemMessagePresets),
    ("INSTRUCT_PROMPT_STYLE_", preset_nodes.UC_InstructPromptPresets),
    ("BONUS_PROMPT_STYLE_", preset_nodes.UC_BonusPromptPresets),
)

RENAMED_EXISTING_SUFFIXES = (
    "PHOTOGRAPHY_PHOTOREALISM",
    "PHOTOGRAPHY_ANALOG_FILM",
    "PHOTOGRAPHY_EARLY_2000S_ANALOG",
    "PHOTOGRAPHY_POLAROID",
    "PHOTOGRAPHY_KODACHROME",
)

NEW_SUFFIXES = (
    "PHOTOGRAPHY_FULL_FRAME_DIGITAL",
    "PHOTOGRAPHY_MEDIUM_FORMAT",
    "PHOTOGRAPHY_CCD_COMPACT_DIGITAL",
    "PHOTOGRAPHY_SMARTPHONE",
    "PHOTOGRAPHY_STUDIO_STROBE",
    "PHOTOGRAPHY_DIRECT_FLASH",
    "PHOTOGRAPHY_AVAILABLE_LIGHT",
    "PHOTOGRAPHY_FINE_ART_BLACK_AND_WHITE",
)

REMOVED_PRESET_IDS = {
    "STYLE_PHOTOREALISM",
    "STYLE_ANALOG_FILM",
    "STYLE_EARLY_2000S_ANALOG",
    "STYLE_POLAROID",
    "STYLE_KODACHROME",
    "STYLE_CONTEMPORARY_DIGITAL_PHOTOGRAPHY",
    "STYLE_DOCUMENTARY_PHOTOGRAPHY",
    "STYLE_EDITORIAL_PHOTOGRAPHY",
    "STYLE_STUDIO_PHOTOGRAPHY",
    "STYLE_DIRECT_FLASH_PHOTOGRAPHY",
    "STYLE_SMARTPHONE_PHOTOGRAPHY",
    "STYLE_COMPACT_DIGITAL_CAMERA",
    "STYLE_MEDIUM_FORMAT_PHOTOGRAPHY",
    "STYLE_MACRO_PHOTOGRAPHY",
    "STYLE_BLACK_AND_WHITE_PHOTOGRAPHY",
}

UNCHANGED_TEXT_HASHES = {
    "SYSTEM_MESSAGE_STYLE_PHOTOGRAPHY_PHOTOREALISM": "4ab4c811ab66e0f595dde4af7f47be65eaba21f62abc9db32a65749410436d8a",
    "SYSTEM_MESSAGE_STYLE_PHOTOGRAPHY_POLAROID": "9a946e97929510433c75c6fea6a150e1402f21962323baf5512dedc078e7bc13",
    "SYSTEM_MESSAGE_STYLE_PHOTOGRAPHY_KODACHROME": "6c5f4ec5f72300ce18da7787bb845a5bc5c17faacb43c51c27412d29242eec53",
    "INSTRUCT_PROMPT_STYLE_PHOTOGRAPHY_PHOTOREALISM": "75a98d69cf6fbf54cc96cee29bf71b015c5004386170e106a7dda30325b15c97",
    "INSTRUCT_PROMPT_STYLE_PHOTOGRAPHY_POLAROID": "cc7723a6e662bf64b85882d9294bb18bd09e643fb29b7997234267b8e05563f5",
    "INSTRUCT_PROMPT_STYLE_PHOTOGRAPHY_KODACHROME": "34064007fc7b4c401a6b88524b0ff645f99b73b9a9024d3b998e364738fe5da1",
    "BONUS_PROMPT_STYLE_PHOTOGRAPHY_PHOTOREALISM": "59558a280631f509e62c7bbb2eae049cd5f19ec8b149b2d172a75a480f5752e6",
    "BONUS_PROMPT_STYLE_PHOTOGRAPHY_POLAROID": "ccfe58ad7193458c4f3e823a36e15d29c8e37b9cef7f7921223c6e93db1f796f",
    "BONUS_PROMPT_STYLE_PHOTOGRAPHY_KODACHROME": "7157bc7de763fa9b69e9b7b99f950eb81dac80d302f440cc6ea61ba14be7c45b",
}

PROCESS_SIGNATURES = {
    "PHOTOGRAPHY_ANALOG_FILM": (
        "color-negative",
        "grain",
        "highlight",
        "halation",
        "shadows",
        "color separation",
        "microcontrast",
        "focus falloff",
    ),
    "PHOTOGRAPHY_EARLY_2000S_ANALOG": (
        "point-and-shoot",
        "minilab",
        "midtones",
        "grain",
        "compact-lens",
        "automatic-exposure",
        "highlight latitude",
    ),
    "PHOTOGRAPHY_FULL_FRAME_DIGITAL": (
        "raw",
        "dynamic range",
        "highlight",
        "shadow",
        "sensor",
        "microcontrast",
        "sharpen",
        "focus",
    ),
    "PHOTOGRAPHY_MEDIUM_FORMAT": (
        "tonal",
        "color",
        "highlight",
        "fine texture",
        "microcontrast",
        "focus falloff",
    ),
    "PHOTOGRAPHY_CCD_COMPACT_DIGITAL": (
        "ccd",
        "jpeg",
        "primary colors",
        "midtones",
        "small-sensor",
        "highlight clipping",
        "chroma",
        "lens",
    ),
    "PHOTOGRAPHY_SMARTPHONE": (
        "multi-frame",
        "tone mapping",
        "computational focus",
        "shadows",
        "highlights",
        "detail enhancement",
        "noise reduction",
    ),
    "PHOTOGRAPHY_STUDIO_STROBE": (
        "strobe",
        "key",
        "fill",
        "specular",
        "shadow",
        "white balance",
        "detail",
        "ambient",
    ),
    "PHOTOGRAPHY_DIRECT_FLASH": (
        "flash",
        "falloff",
        "hard-edged shadows",
        "foreground",
        "ambient",
        "specular",
        "contrast",
    ),
    "PHOTOGRAPHY_AVAILABLE_LIGHT": (
        "ambient",
        "color temperature",
        "aperture",
        "focus falloff",
        "shadow",
        "highlight",
        "high-iso",
    ),
    "PHOTOGRAPHY_FINE_ART_BLACK_AND_WHITE": (
        "panchromatic",
        "luminance",
        "tonal",
        "midtones",
        "blacks",
        "highlight",
        "monochrome",
    ),
}


def _value(prefix, suffix):
    return getattr(presets_collection, f"{prefix}{suffix}")


def test_photography_choices_are_grouped_and_shared():
    expected = {
        f"STYLE_{suffix}"
        for suffix in RENAMED_EXISTING_SUFFIXES + NEW_SUFFIXES
    }
    shared = set(preset_nodes.UC_UnifiedPresets.get_shared_presets())

    assert expected <= shared
    for _, node in FAMILIES:
        choices = set(node.get_presets())
        assert expected <= choices
        assert choices.isdisjoint(REMOVED_PRESET_IDS)


def test_unchanged_photography_text_is_byte_identical():
    actual = {
        name: hashlib.sha256(getattr(presets_collection, name).encode()).hexdigest()
        for name in UNCHANGED_TEXT_HASHES
    }
    assert actual == UNCHANGED_TEXT_HASHES


def test_camera_shot_catalog_is_unchanged():
    values = {
        name: value
        for name, value in vars(presets_collection).items()
        if name.startswith("INSTRUCT_PROMPT_CAMERA_SHOT_")
    }
    payload = "\n".join(f"{name}={values[name]}" for name in sorted(values))
    assert len(values) == 20
    assert hashlib.sha256(payload.encode()).hexdigest() == (
        "1cdbff162b2f166b3cb0cbd470c93d8ba46468476ad6edef84dcd683dac5ba96"
    )


def test_process_signatures_are_deep_and_distinct():
    for suffix, required_terms in PROCESS_SIGNATURES.items():
        components = [_value(prefix, suffix) for prefix, _ in FAMILIES]
        combined = " ".join(components).lower()
        assert all(term in combined for term in required_terms), suffix
        assert len(set(components)) == len(components)


def test_component_roles_are_complementary():
    for suffix in PROCESS_SIGNATURES:
        system = _value("SYSTEM_MESSAGE_STYLE_", suffix)
        instruct = _value("INSTRUCT_PROMPT_STYLE_", suffix)
        bonus = _value("BONUS_PROMPT_STYLE_", suffix)

        assert system.startswith("You are an AI that specializes in writing image-editing descriptions")
        assert "one concise structured instruction" in system
        assert instruct.startswith("Render the reference as")
        assert "Preserve the composition" in instruct or "preserve the composition" in instruct
        assert "Preserve the" not in bonus
        assert len(system) > len(instruct) > len(bonus)


def test_process_presets_do_not_reframe_or_invent_content():
    banned = (
        "close-up",
        "wide shot",
        "camera angle",
        "magnif",
        "reframe",
        "crop the",
        "add a",
        "add an",
        "subject",
        "person",
        "animal",
        "portrait",
        "product",
    )
    for suffix in PROCESS_SIGNATURES:
        for prefix, _ in FAMILIES:
            normalized = _value(prefix, suffix).lower()
            assert all(term not in normalized for term in banned), (suffix, prefix)


def test_analog_presets_do_not_stack_distressed_media_artifacts():
    banned = (
        "heavy grain",
        "light leak",
        "scratch",
        "dust",
        "faded",
        "damage",
        "grainy and raw",
        "imperfection",
    )
    for suffix in ("PHOTOGRAPHY_ANALOG_FILM", "PHOTOGRAPHY_EARLY_2000S_ANALOG"):
        combined = " ".join(_value(prefix, suffix) for prefix, _ in FAMILIES).lower()
        assert all(term not in combined for term in banned)


def test_preset_execution_returns_exact_stored_text():
    for suffix in RENAMED_EXISTING_SUFFIXES + NEW_SUFFIXES:
        preset_id = f"STYLE_{suffix}"
        for prefix, node in FAMILIES:
            stored = _value(prefix, suffix)
            assert isinstance(stored, str)
            assert node.execute(preset_id).args[0] == stored
