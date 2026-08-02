import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


RUNNER_PATH = Path(__file__).with_name("run_tests.py")
SPEC = importlib.util.spec_from_file_location("utils_collection_test_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_manifest_maps_every_tracked_production_source():
    groups = runner.load_groups()
    tracked = runner.git_lines("ls-files", "*.py", "web/*.js")
    production = {path for path in tracked if runner.is_production_source(path)}

    selection = runner.select_tests(production, groups)

    assert selection.unmapped == set()


def test_changed_paths_select_only_dependent_groups_and_direct_tests():
    groups = runner.load_groups()
    selection = runner.select_tests(
        {
            "staged_compositor_helpers.py",
            "encoder_helpers.py",
            "tests/test_scheduler_migration.py",
            "README.md",
        },
        groups,
    )

    assert selection.groups == {"composite", "encoder", "textgen"}
    assert "tests/test_composite_nodes.py" in selection.python_tests
    assert "tests/test_advanced_visual_consensus.py" in selection.python_tests
    assert "tests/test_textgen_formula.py" in selection.python_tests
    assert "tests/test_scheduler_migration.py" in selection.python_tests
    assert selection.frontend_tests == set()


def test_frontend_source_selects_frontend_and_parity_coverage():
    selection = runner.select_tests(
        {"web/layered_background_editor.js"}, runner.load_groups()
    )

    assert selection.groups == {"staged_frontend"}
    assert "tests/test_composite_nodes.py" in selection.python_tests
    assert "tests/test_layered_placement.mjs" in selection.frontend_tests
    assert "tests/test_staged_editor_layout.mjs" in selection.frontend_tests


def test_unknown_production_source_fails_closed():
    selection = runner.select_tests({"new_domain.py"}, runner.load_groups())

    assert selection.unmapped == {"new_domain.py"}


def test_explicit_group_and_unknown_group_behavior():
    groups = runner.load_groups()
    selection = runner.select_tests(set(), groups, ("tiling",))

    assert selection.groups == {"tiling"}
    assert selection.python_tests == {
        "tests/test_high_resolution_tiling.py",
        "tests/test_high_resolution_tiling_guide.py",
    }
    with pytest.raises(ValueError, match="Unknown test group"):
        runner.select_tests(set(), groups, ("missing",))


def test_final_test_discovery_excludes_untracked_tests(monkeypatch):
    monkeypatch.setattr(
        runner,
        "git_lines",
        lambda *args: {
            "tests/test_one.py",
            "tests/test_two.mjs",
        },
    )

    groups = {
        "configured": runner.TestGroup(
            "configured",
            (),
            ("tests/test_configured.py",),
            ("tests/test_configured.mjs",),
        )
    }

    assert runner.tracked_final_tests(groups) == (
        {"tests/test_one.py", "tests/test_configured.py"},
        {"tests/test_two.mjs", "tests/test_configured.mjs"},
    )


def test_run_selection_uses_configured_interpreter_and_cleans_temp(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runner, "TEMP_ROOT", tmp_path / "pytest-temp")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs))
        or subprocess.CompletedProcess(command, 0),
    )
    selection = runner.Selection(
        python_tests={"tests/test_bounding_box.py"},
        frontend_tests={"tests/test_layered_placement.mjs"},
    )

    assert runner.run_selection(selection) == 0
    assert calls[0][0][:4] == [runner.sys.executable, "-m", "pytest", "-q"]
    assert calls[0][1]["cwd"] == runner.COMFYUI_ROOT
    assert calls[1][0][:2] == ["node", "--test"]
    assert not runner.TEMP_ROOT.exists()
