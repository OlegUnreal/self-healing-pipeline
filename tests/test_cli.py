from self_healing.cli import build_parser, main


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


def test_cli_demo_tools(capsys):
    code = main(["--demo-tools"])
    out = capsys.readouterr().out
    assert code == 0
    assert "success: True" in out
