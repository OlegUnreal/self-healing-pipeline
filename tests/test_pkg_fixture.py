from self_healing.demo_tools import (
    PKG_APP,
    PKG_TEST,
    PKG_UTIL_FIXED,
    detect_scripted_fix,
    make_scripted_planner,
    write_pkg_fixture,
)
from self_healing.agent import heal_with_tools
from self_healing.graph import heal_with_graph
from self_healing.tools import Workspace
from self_healing.verifier import verify


def test_the_scripted_repair_satisfies_the_fixture(tmp_path):
    """Broken before: the demo asserted an unescaped result while its scripted fix
    added `from html import escape`, so no repair could ever turn this workspace
    green and every `--src <dir>` demo reported failure. The pair must agree."""
    write_pkg_fixture(tmp_path)
    (tmp_path / "util.py").write_text(PKG_UTIL_FIXED)
    assert verify(PKG_TEST, cwd=tmp_path).exit_code == 0


def test_pkg_fixture_fails_as_name_error(tmp_path):
    write_pkg_fixture(tmp_path)
    result = verify(PKG_TEST, cwd=tmp_path)
    assert result.exit_code != 0
    assert result.error == "name_error"


def test_detect_pkg_fixture(tmp_path):
    write_pkg_fixture(tmp_path)
    fix = detect_scripted_fix(tmp_path)
    assert fix is not None
    assert fix.patch_path == "util.py"


def test_tools_heal_pkg_workspace(tmp_path):
    write_pkg_fixture(tmp_path)
    src = tmp_path / "app.py"
    planner = make_scripted_planner(detect_scripted_fix(tmp_path), test_code=PKG_TEST)
    report = heal_with_tools(src, PKG_TEST, planner, max_steps=8, workspace=Workspace(tmp_path))
    assert report.success is True
    assert "from html import escape" in (tmp_path / "util.py").read_text()
    assert (tmp_path / "app.py").read_text() == PKG_APP
    assert "apply_patch" in [t["tool"] for t in report.tool_trace]


def test_graph_heal_pkg_workspace(tmp_path):
    write_pkg_fixture(tmp_path)
    src = tmp_path / "app.py"
    planner = make_scripted_planner(detect_scripted_fix(tmp_path), test_code=PKG_TEST)
    report = heal_with_graph(src, PKG_TEST, planner, max_steps=8, workspace=Workspace(tmp_path))
    assert report.success is True
    assert "from html import escape" in (tmp_path / "util.py").read_text()
