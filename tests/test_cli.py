from self_healing.cli import build_parser, main
from self_healing.demo_tools import write_pkg_fixture


def test_help_without_args():
    assert main([]) == 2


def test_parser_accepts_workspace_flags():
    args = build_parser().parse_args(
        ["--src", "examples/add.py", "--test", "examples/test_add.py", "--mode", "graph", "--stream"]
    )
    assert args.mode == "graph"
    assert args.stream is True


def test_cli_repairs_example_with_stub_graph(tmp_path, capsys):
    src = tmp_path / "add.py"
    src.write_text("def add(a, b):\n    return a - b  # bug: should be +\n")
    test = tmp_path / "test_add.py"
    test.write_text("from add import add\nassert add(2, 3) == 5\nprint(\"ok\")\n")
    code = main(
        ["--src", str(src), "--test", str(test), "--mode", "graph", "--planner", "stub", "--stream"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "success: True" in out
    assert "return a + b" in src.read_text()


def test_cli_repairs_directory_workspace(tmp_path, capsys):
    write_pkg_fixture(tmp_path)
    code = main(
        [
            "--src",
            str(tmp_path),
            "--test",
            str(tmp_path / "test_app.py"),
            "--mode",
            "tools",
            "--planner",
            "stub",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "success: True" in out
    assert "from html import escape" in (tmp_path / "util.py").read_text()


def test_cli_stub_unknown_workspace(tmp_path, capsys):
    src = tmp_path / "mystery.py"
    src.write_text("def f():\n    return 1\n")
    test = tmp_path / "test_mystery.py"
    test.write_text("from mystery import f\nassert f() == 2\n")
    code = main(["--src", str(src), "--test", str(test), "--planner", "stub"])
    err = capsys.readouterr().err
    assert code == 2
    assert "no fixture" in err


def test_cli_demo_tools(capsys):
    code = main(["--demo-tools"])
    out = capsys.readouterr().out
    assert code == 0
    assert "success: True" in out
