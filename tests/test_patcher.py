from pathlib import Path

from self_healing.patcher import apply_diff


def test_apply_and_rollback(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("x = 1\n")
    ok = apply_diff(f, "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n")
    assert ok
    assert f.read_text() == "x = 2\n"
