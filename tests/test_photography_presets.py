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
    "SYSTEM_MESSAGE_STYLE_PHOTOGRAPHY_ANALOG_FILM": "79652a3a6ed35f980e1c69326e00ba2935d2c84e78061b5afe19b9ca4c41633a",
    "SYSTEM_MESSAGE_STYLE_PHOTOGRAPHY_EARLY_2000S_ANALOG": "d3dcee2c91340bc6d1031373493ae59c10ea1801682ccb64e26b1e767b72d9ec",
    "SYSTEM_MESSAGE_STYLE_PHOTOGRAPHY_POLAROID": "9a946e97929510433c75c6fea6a150e1402f21962323baf5512dedc078e7bc13",
    "SYSTEM_MESSAGE_STYLE_PHOTOGRAPHY_KODACHROME": "6c5f4ec5f72300ce18da7787bb845a5bc5c17faacb43c51c27412d29242eec53",
    "INSTRUCT_PROMPT_STYLE_PHOTOGRAPHY_PHOTOREALISM": "75a98d69cf6fbf54cc96cee29bf71b015c5004386170e106a7dda30325b15c97",
    "INSTRUCT_PROMPT_STYLE_PHOTOGRAPHY_ANALOG_FILM": "acd1e383779176628b7da162909357115da5aa89df532bafd3709148bbd964cd",
    "INSTRUCT_PROMPT_STYLE_PHOTOGRAPHY_EARLY_2000S_ANALOG": "1bcba802ccb7ba81934663f45cc191f784d7d862c5f93bd38cd72d199542dbe6",
    "INSTRUCT_PROMPT_STYLE_PHOTOGRAPHY_POLAROID": "cc7723a6e662bf64b85882d9294bb18bd09e643fb29b7997234267b8e05563f5",
    "INSTRUCT_PROMPT_STYLE_PHOTOGRAPHY_KODACHROME": "34064007fc7b4c401a6b88524b0ff645f99b73b9a9024d3b998e364738fe5da1",
    "BONUS_PROMPT_STYLE_PHOTOGRAPHY_PHOTOREALISM": "59558a280631f509e62c7bbb2eae049cd5f19ec8b149b2d172a75a480f5752e6",
    "BONUS_PROMPT_STYLE_PHOTOGRAPHY_ANALOG_FILM": "7ae0fd9fc52f0eb0dd611382034c13b07f3ab6e73a3f08d4ee3da013a48dc484",
    "BONUS_PROMPT_STYLE_PHOTOGRAPHY_EARLY_2000S_ANALOG": "d78ed180c4fb065b71cbaa4f4ac32e1e5ce0e9b6ab686839be178abc495890bd",
    "BONUS_PROMPT_STYLE_PHOTOGRAPHY_POLAROID": "ccfe58ad7193458c4f3e823a36e15d29c8e37b9cef7f7921223c6e93db1f796f",
    "BONUS_PROMPT_STYLE_PHOTOGRAPHY_KODACHROME": "7157bc7de763fa9b69e9b7b99f950eb81dac80d302f440cc6ea61ba14be7c45b",
}

VISIBLE_PHOTOGRAPHIC_SIGNATURES = {
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
        "aperture",
        "depth of field",
        "exposure",
        "highlight",
        "shadow",
        "low iso",
        "texture",
        "light",
    ),
    "PHOTOGRAPHY_MEDIUM_FORMAT": (
        "wide aperture",
        "shallow depth of field",
        "bokeh",
        "tonal separation",
        "highlight",
        "color gradation",
        "low iso",
        "texture",
    ),
    "PHOTOGRAPHY_CCD_COMPACT_DIGITAL": (
        "short focal length",
        "small aperture",
        "deep depth of field",
        "automatic exposure",
        "direct color",
        "local contrast",
        "clipped",
        "chroma noise",
        "edge",
    ),
    "PHOTOGRAPHY_SMARTPHONE": (
        "wide angle",
        "deep depth of field",
        "automatic exposure",
        "lifted shadow",
        "controlled highlight",
        "edge sharpening",
        "low noise",
        "natural color",
    ),
    "PHOTOGRAPHY_STUDIO_STROBE": (
        "key light",
        "fill",
        "specular",
        "shadow",
        "white balance",
        "moderate aperture",
        "sharp",
        "low iso",
        "ambient",
    ),
    "PHOTOGRAPHY_DIRECT_FLASH": (
        "frontal flash",
        "hard edged shadows",
        "falloff",
        "near plane",
        "darker",
        "specular",
        "moderate aperture",
        "frozen motion",
        "contrast",
    ),
    "PHOTOGRAPHY_AVAILABLE_LIGHT": (
        "existing",
        "color temperature",
        "wide aperture",
        "shallow depth of field",
        "exposure",
        "shadow",
        "highlight rolloff",
        "high iso",
        "grain",
    ),
    "PHOTOGRAPHY_FINE_ART_BLACK_AND_WHITE": (
        "black-and-white",
        "grayscale",
        "no residual color",
        "tonal separation",
        "midtones",
        "blacks",
        "highlight rolloff",
        "monochrome grain",
        "focus",
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


def test_photographic_signatures_are_visible_and_distinct():
    for suffix, required_terms in VISIBLE_PHOTOGRAPHIC_SIGNATURES.items():
        components = [_value(prefix, suffix) for prefix, _ in FAMILIES]
        combined = " ".join(components).lower()
        assert all(term in combined for term in required_terms), suffix
        assert len(set(components)) == len(components)


def test_component_roles_are_complementary():
    for suffix in VISIBLE_PHOTOGRAPHIC_SIGNATURES:
        system = _value("SYSTEM_MESSAGE_STYLE_", suffix)
        instruct = _value("INSTRUCT_PROMPT_STYLE_", suffix)
        bonus = _value("BONUS_PROMPT_STYLE_", suffix)

        assert system.startswith("You are an AI that specializes in writing image-editing descriptions")
        assert "one concise structured instruction" in system
        assert instruct.startswith("Render the reference as")
        assert "Preserve the composition" in instruct or "preserve the composition" in instruct
        assert "Preserve the" not in bonus
        assert len(system) > len(instruct) > len(bonus)


def test_photography_presets_do_not_reframe_or_invent_content():
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
    for suffix in VISIBLE_PHOTOGRAPHIC_SIGNATURES:
        for prefix, _ in FAMILIES:
            normalized = _value(prefix, suffix).lower()
            assert all(term not in normalized for term in banned), (suffix, prefix)


def test_smartphone_preset_does_not_name_devices_or_interfaces():
    banned = (
        "smartphone",
        "mobile",
        "camera",
        "screen",
        "interface",
        "viewfinder",
        "overlay",
    )
    for prefix, _ in FAMILIES:
        normalized = _value(prefix, "PHOTOGRAPHY_SMARTPHONE").lower()
        assert all(term not in normalized for term in banned), prefix


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
