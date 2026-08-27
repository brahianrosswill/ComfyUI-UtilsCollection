import ast
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "scripts" / "manage_vlm_preset_authorities.py"
H3_QUERY_TOOL_PATH = ROOT / "scripts" / "sync_vlm_h3_ref2va_query_preset.py"
CONFIG = (
    (
        "vlm_presets.py",
        "vlm_presets_vars.py",
        (
            "system_instructions_vlm",
            "system_query_additional_vlm",
            "system_query_raw_vlm",
            "additional_instructions_vlm",
        ),
        (),
    ),
    (
        "vlm_experimental_presets.py",
        "vlm_experimental_presets_vars.py",
        ("system_instructions_vlm_experimental",),
        (),
    ),
    (
        "minimax_h3_vlm_presets.py",
        "minimax_h3_vlm_presets_vars.py",
        ("minimax_h3_system_instructions_vlm",),
        ("minimax_h3_vlm_jailbreak_prefix", "minimax_h3_vlm_jailbreak_suffix"),
    ),
    (
        "minimax_h3_vlm_experimental_presets.py",
        "minimax_h3_vlm_experimental_presets_vars.py",
        ("minimax_h3_system_instructions_vlm_experimental",),
        (),
    ),
)


def assert_crlf_only(label, value):
    if "\n" not in value and "\r" not in value:
        return
    assert "\r\n" in value, f"{label} has no CRLF newline"
    remainder = value.replace("\r\n", "")
    assert "\n" not in remainder, f"{label} contains LF-only newlines"
    assert "\r" not in remainder, f"{label} contains lone CR newlines"


def test_nonlegacy_vlm_authorities_match_runtime_and_use_crlf():
    for runtime_name, authority_name, dictionaries, scalars in CONFIG:
        runtime = runpy.run_path(str(ROOT / runtime_name))
        authority = runpy.run_path(str(ROOT / authority_name))
        authoritative_dictionaries = authority["RUNTIME_DICTIONARIES"]
        authoritative_values = authority["RUNTIME_VALUES"]

        assert tuple(authoritative_dictionaries) == dictionaries
        assert tuple(authoritative_values) == scalars
        for dictionary_name in dictionaries:
            expected = authoritative_dictionaries[dictionary_name]
            actual = runtime[dictionary_name]
            assert tuple(actual) == tuple(expected)
            assert actual == expected
            for key, value in expected.items():
                assert_crlf_only(f"{authority_name}:{dictionary_name}.{key}", value)
                assert_crlf_only(f"{runtime_name}:{dictionary_name}.{key}", actual[key])
        for scalar_name in scalars:
            assert runtime[scalar_name] == authoritative_values[scalar_name]
            assert_crlf_only(
                f"{authority_name}:{scalar_name}", authoritative_values[scalar_name]
            )
            assert_crlf_only(f"{runtime_name}:{scalar_name}", runtime[scalar_name])


def test_h3_raw_query_authorities_are_derived_from_prefix_and_suffix():
    source = (ROOT / "vlm_presets_vars.py").read_text(encoding="utf-8")
    assignments = {
        target.id: node.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    expected = {
        "H3_REF2VA": ("H3_REF2VA_PREFIX", "H3_REF2VA_SUFFIX"),
        "H3_REF2VA_ALT": ("H3_REF2VA_ALT_PREFIX", "H3_REF2VA_ALT_SUFFIX"),
        "H3_MIXED_REF2VA": (
            "H3_MIXED_REF2VA_PREFIX",
            "H3_MIXED_REF2VA_SUFFIX",
        ),
    }

    for name, arguments in expected.items():
        value = assignments[name]
        assert isinstance(value, ast.Call)
        assert isinstance(value.func, ast.Name)
        assert value.func.id == "_h3_raw"
        assert tuple(arg.id for arg in value.args if isinstance(arg, ast.Name)) == arguments


def test_tracked_h3_query_synchronizer_reports_synchronized_state():
    result = subprocess.run(
        [sys.executable, str(H3_QUERY_TOOL_PATH)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        "Runtime H3 reference query presets are already synchronized."
    )


def test_authority_tool_dry_run_and_apply_changed_content(tmp_path, capsys):
    runtime_path = tmp_path / "runtime.py"
    authority_path = tmp_path / "authority.py"
    runtime_path.write_text('sample = {"entry": "old\\r\\nvalue"}\n', encoding="utf-8")
    authority_path.write_text(
        'RUNTIME_DICTIONARIES = {"sample": {"entry": "new\\r\\nvalue"}}\n'
        "RUNTIME_VALUES = {}\n",
        encoding="utf-8",
    )

    tool = runpy.run_path(str(TOOL_PATH))
    main = tool["main"]
    main.__globals__["ROOT"] = tmp_path
    main.__globals__["CONFIG"] = (("runtime.py", "authority.py", ("sample",), ()),)

    original = runtime_path.read_bytes()
    main([])
    report = json.loads(capsys.readouterr().out)
    assert report["changed"] == ["runtime.py"]
    assert runtime_path.read_bytes() == original

    main(["--apply"])
    capsys.readouterr()
    source = runtime_path.read_bytes()
    assert b"\r\n" in source
    assert b"\n" not in source.replace(b"\r\n", b"")
    assert runpy.run_path(str(runtime_path))["sample"] == {"entry": "new\r\nvalue"}


def test_authority_tool_preservation_guard_rejects_content_changes():
    validate_preserved = runpy.run_path(str(TOOL_PATH))["validate_preserved"]
    runtime = {"sample": {"entry": "old\r\nvalue"}}
    authority = {
        "RUNTIME_DICTIONARIES": {"sample": {"entry": "new\r\nvalue"}},
        "RUNTIME_VALUES": {},
    }

    with pytest.raises(RuntimeError, match="Non-newline content changed"):
        validate_preserved(runtime, authority, ("sample",), ())
