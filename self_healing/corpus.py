"""Repair-corpus seed data used to fit the embedding and classifier models.

Each entry is a real-shaped pytest/traceback transcript labelled with the class the
existing rule verifier *should* report. Several entries are deliberately ambiguous:
the raw output contains more than one exception and only the root-cause frame decides
the label, which is exactly where keyword rules break down.
"""
from __future__ import annotations

from typing import Final

LABELS: Final[tuple[str, ...]] = (
    "assertion_failure",
    "import_error",
    "indentation_error",
    "name_error",
    "syntax_error",
    "timeout",
    "type_error",
    "runtime_error",
)


def _tb(frames: str, exc: str) -> str:
    return f"================ test session starts =================\n{frames}E       {exc}\n1 failed in 0.42s\n"


_ASSERT = (
    _tb(
        'def test_rounds_up():\n>       assert ceil_cents(1.005) == 2, "half-up rounding"\nE       AssertionError: half-up rounding\nE       assert 1 == 2\n',
        "AssertionError: half-up rounding",
    ),
    _tb(
        "def test_split_even():\n>       assert allocate(10, 3) == [4, 3, 3]\nE       assert [3, 3, 4] == [4, 3, 3]\nE         At index 0 diff: 3 != 4\n",
        "AssertionError",
    ),
)


def _build() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []

    def add(text: str, label: str) -> None:
        rows.append((text, label))

    for text in _ASSERT:
        add(text, "assertion_failure")
    add(
        _tb(
            "def test_cache_ttl():\n>       assert entry.expires_at > now\nE       assert 1719.0 > 1720.5\n",
            "AssertionError: ttl window expired",
        ),
        "assertion_failure",
    )
    add(
        _tb(
            "def test_token():\n>       assert decode(tok) == {'sub': 'u1'}\nE       AssertionError: payload mismatch\n",
            "AssertionError",
        ),
        "assertion_failure",
    )
    add(
        _tb(
            "def test_reconcile():\n>       assert ledger.balance == 0\nE       assert -12 == 0\n",
            "AssertionError",
        ),
        "assertion_failure",
    )
    add(
        _tb(
            "def test_dedupe():\n>       assert len(rows) == 3\nE       assert 5 == 3\nE        +  where 5 = len(deduped)\n",
            "AssertionError",
        ),
        "assertion_failure",
    )

    for mod in ("pandas", "numpy", "rich", "pytest_asyncio", "ujson", "yaml", "orjson", "httpx"):
        add(
            _tb(
                f"def test_import():\n    import {mod}\nE   ModuleNotFoundError: No module named {mod!r}\n",
                f"ModuleNotFoundError: No module named {mod!r}",
            ),
            "import_error",
        )
    add(
        _tb(
            "    from .helpers import parse  # noqa\nE   ImportError: cannot import name 'parse' from 'tests.helpers'",
            "ImportError: cannot import name 'parse' from 'tests.helpers'",
        ),
        "import_error",
    )
    add(
        _tb(
            "    import self_healing.memory\nE   ModuleNotFoundError: No module named 'self_healing.memory'; 'self_healing' is not a package",
            "ModuleNotFoundError",
        ),
        "import_error",
    )
    add(
        _tb(
            "    importlib.import_module(plugin_name)\nE   ImportError: dlopen(): symbol _pystartup not found",
            "ImportError",
        ),
        "import_error",
    )
    add(
        _tb(
            "    raise ImportError(msg) from e\nE   ImportError: tokenizers>=0.14,<0.20 is required",
            "ImportError: tokenizers>=0.14,<0.20 is required",
        ),
        "import_error",
    )

    for i in range(6):
        add(
            _tb(
                f"    def heal(x):\n  return x + {i}      # off by one block\nE       IndentationError: unexpected indent",
                "IndentationError: unexpected indent",
            ),
            "indentation_error",
        )
    add(
        _tb(
            "    if ok:\n        pass\n      else:\nE       IndentationError: unindent does not match any outer indentation level",
            "IndentationError",
        ),
        "indentation_error",
    )
    add(
        _tb(
            "    def apply_patch(\n  path, diff):\nE       IndentationError: expected an indented block after function definition",
            "IndentationError",
        ),
        "indentation_error",
    )
    add(
        _tb(
            "    for row in rows:\n    yield row\nE       IndentationError: expected an indented block",
            "IndentationError",
        ),
        "indentation_error",
    )
    add(
        _tb(
            "    try:\n    result = run()\nE       IndentationError: expected an indented block after 'try' statement",
            "IndentationError",
        ),
        "indentation_error",
    )
    add(
        _tb(
            "    while pending:\n        pop()\n      pending = []\nE       IndentationError: unindent does not match any outer indentation level",
            "IndentationError",
        ),
        "indentation_error",
    )

    for name in ("registry", "Workspace", "SETTINGS", "propose_diff", "tool_specs", "hits"):
        add(
            _tb(
                f"    value = {name}.lookup(key)\nE   NameError: name '{name}' is not defined",
                f"NameError: name '{name}' is not defined",
            ),
            "name_error",
        )
    add(
        _tb(
            "    return _classify(res)\nE   NameError: name '_classify' is not defined. Did you mean: 'classify_output'?",
            "NameError",
        ),
        "name_error",
    )
    add(
        _tb(
            "    assert heal.__version__\nE   NameError: name '__version__' is not defined",
            "NameError",
        ),
        "name_error",
    )
    add(
        _tb(
            "    spec = ToolSpec(**row)\nE   NameError: name 'ToolSpec' is not defined",
            "NameError",
        ),
        "name_error",
    )

    for src, msg in (
        ("def heal(:\n    pass", "syntax_error"),
        ("result = [1, 2\nE   SyntaxError: '[' was never closed", "syntax_error"),
        ("if ok else False\nE   SyntaxError: invalid syntax", "syntax_error"),
        ("return *args*\nE   SyntaxError: can't use star here", "syntax_error"),
        ("def f() {{ }}\nE   SyntaxError: invalid syntax", "syntax_error"),
        ("x = = 3\nE   SyntaxError: invalid syntax", "syntax_error"),
    ):
        add(_tb(f"    {src}\nE   SyntaxError: {msg}", "SyntaxError"), "syntax_error")
    add(
        _tb(
            "    eval(compile(src, '<heal>', 'eval'))\nE   SyntaxError: unexpected EOF while parsing",
            "SyntaxError",
        ),
        "syntax_error",
    )
    add(
        _tb(
            "    importlib.reload(module)\nE   SyntaxError: source code string cannot contain null bytes",
            "SyntaxError",
        ),
        "syntax_error",
    )

    add(_tb("    wait_for(test_run)\nE   TimeoutError: run_tests exceeded 5.0s wall budget", "TimeoutError"), "timeout")
    add(
        _tb(
            "    sandbox.run_code(src, timeout=timeout)\nE   TimeoutExpired: timed out after 5 seconds while running pytest",
            "subprocess.TimeoutExpired",
        ),
        "timeout",
    )
    add(_tb("    drain(pipe)\nE   TimeoutError: the read operation timed out", "TimeoutError"), "timeout")
    add(
        _tb(
            "    while not done:\n        pass  # spin: no progress, killed by rlimit watchdog\nE   TimeoutError: sandbox exceeded RLIMIT_AS and was killed",
            "TimeoutError",
        ),
        "timeout",
    )
    add(
        _tb(
            "    pytest.main(['-q'])\nE   TimeoutError: test run produced no output for 30s (deadlock in fixture teardown)",
            "TimeoutError",
        ),
        "timeout",
    )
    add(
        _tb(
            "    await asyncio.sleep(3600)\nE   TimeoutError: coroutine did not yield; event loop budget exhausted",
            "TimeoutError",
        ),
        "timeout",
    )

    for i in range(6):
        add(
            _tb(
                f"    total = amount + '1'\nE   TypeError: unsupported operand type(s) for +: 'int' and 'str'  # variant {i}",
                "TypeError",
            ),
            "type_error",
        )
    add(
        _tb(
            "    return len(None)\nE   TypeError: object of type 'NoneType' has no len()",
            "TypeError",
        ),
        "type_error",
    )
    add(
        _tb(
            "    row.update(**args)\nE   TypeError: update() takes no keyword arguments",
            "TypeError",
        ),
        "type_error",
    )
    add(
        _tb(
            "    diff = patcher.apply_diff(target, 42)\nE   TypeError: expected str, bytes or os.PathLike, not int",
            "TypeError",
        ),
        "type_error",
    )
    add(
        _tb(
            "    return sorted(hits, key=lambda h: h['score'], reverse)\nE   TypeError: 'builtin_function_or_method' object is not iterable",
            "TypeError",
        ),
        "type_error",
    )
    add(
        _tb(
            "    return f'{value:,.2%}' if value else ''\nE   TypeError: unsupported format string passed to NoneType.__format__",
            "TypeError",
        ),
        "type_error",
    )

    for i in range(5):
        add(
            _tb(
                f"    row = mapping[key]\nE   KeyError: 'row-{i}'\nE   \nE   The above exception was the direct cause of the following exception:\nE   RuntimeError: index build aborted",
                "KeyError",
            ),
            "runtime_error",
        )
    add(
        _tb(
            "    raise RuntimeError('workspace escaped jail root')\nE   RuntimeError: workspace escaped jail root",
            "RuntimeError",
        ),
        "runtime_error",
    )
    add(
        _tb(
            "    return 1 / 0\nE   ZeroDivisionError: division by zero",
            "ZeroDivisionError",
        ),
        "runtime_error",
    )
    add(
        _tb(
            "    raise ValueError('empty diff')\nE   ValueError: empty diff",
            "ValueError",
        ),
        "runtime_error",
    )
    add(
        _tb(
            "    conn.execute(sql)\nE   sqlite3.OperationalError: database is locked",
            "sqlite3.OperationalError",
        ),
        "runtime_error",
    )

    # Ambiguous transcripts: rules see the first exception token, root cause is another.
    add(
        _tb(
            "    setup = load_fixture(name)\n"
            "E   KeyError: 'expected'\n"
            "E   \n"
            "E   During handling of the above exception, another exception occurred:\n"
            "E   \n"
            "    assert ceil_cents(x) == 2\n"
            "E   AssertionError: expected rounding to be half-up\n"
            "E   assert 1 == 2",
            "assertion_failure",
        ),
        "assertion_failure",
    )
    add(
        _tb(
            "    import ujson  # optional fast path\n"
            "E   ModuleNotFoundError: No module named 'ujson'\n"
            "E   \n"
            "E   The above exception was the direct cause of the following exception:\n"
            "E   \n"
            "    return json_loads(raw)\n"
            "E   NameError: name 'json_loads' is not defined",
            "name_error",
        ),
        "name_error",
    )
    add(
        _tb(
            "    try:\n        timeout_wait(proc)\n"
            "E   TimeoutError: 5s exceeded\n"
            "E   \n"
            "E   During handling of the above exception, another exception occurred:\n"
            "E   \n"
            "    raise RuntimeError('test suite hung on stdin')\n"
            "E   RuntimeError: test suite hung on stdin",
            "runtime_error",
        ),
        "runtime_error",
    )
    add(
        _tb(
            "    amount = parse(text)\n"
            "E   TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'\n"
            "E   \n"
            "E   The above exception was the direct cause of the following exception:\n"
            "E   \n"
            "    assert rows == expected\n"
            "E   AssertionError: rows differ after currency conversion",
            "assertion_failure",
        ),
        "assertion_failure",
    )
    add(
        _tb(
            "    from .legacy import bridge\n"
            "E   ImportError: attempted relative import with no known parent package\n"
            "E   \n"
            "E   During handling of the above exception, another exception occurred:\n"
            "E   \n"
            "    def fix(:\n"
            "E   SyntaxError: invalid syntax",
            "syntax_error",
        ),
        "syntax_error",
    )
    return rows


SEED_TRACEBACKS: Final[list[tuple[str, str]]] = _build()


def seed_of(label: str, nth: int = 0) -> str:
    """Nth seed transcript of a class, by label rather than by position."""
    matches = [text for text, cls in SEED_TRACEBACKS if cls == label]
    if not matches:
        raise KeyError(f"no seed traceback labelled {label!r}")
    return matches[nth if nth < len(matches) else 0]


REPAIR_EXAMPLES: Final[list[dict[str, object]]] = [
    {
        "failure_class": "assertion_failure",
        "traceback": seed_of("assertion_failure", 1),
        "diff": "--- a/allocate.py\n+++ b/allocate.py\n@@\n-    return [total // parts] * parts\n+    base, extra = divmod(total, parts)\n+    return [base + (i < extra) for i in range(parts)]\n",
        "success": True,
        "attempts": 2,
    },
    {
        "failure_class": "import_error",
        "traceback": seed_of("import_error"),
        "diff": "--- a/requirements.txt\n+++ b/requirements.txt\n@@\n-pytest>=8.0\n+pytest>=8.0\n+pytest-asyncio>=0.23\n",
        "success": True,
        "attempts": 1,
    },
    {
        "failure_class": "type_error",
        "traceback": seed_of("type_error"),
        "diff": "--- a/money.py\n+++ b/money.py\n@@\n-    total = amount + rate\n+    total = amount + Decimal(rate)\n",
        "success": True,
        "attempts": 3,
    },
    {
        "failure_class": "indentation_error",
        "traceback": seed_of("indentation_error"),
        "diff": "--- a/graph.py\n+++ b/graph.py\n@@\n-  return x + 1\n+        return x + 1\n",
        "success": False,
        "attempts": 5,
    },
    {
        "failure_class": "timeout",
        "traceback": seed_of("timeout"),
        "diff": "--- a/sandbox.py\n+++ b/sandbox.py\n@@\n-    while not done:\n-        pass\n+    while not done:\n+        time.sleep(0.01)\n",
        "success": True,
        "attempts": 2,
    },
    {
        "failure_class": "name_error",
        "traceback": seed_of("name_error"),
        "diff": "--- a/handlers.py\n+++ b/handlers.py\n@@\n-    return ToolSpec(**row)\n+    from .registry import ToolSpec\n+\n+    return ToolSpec(**row)\n",
        "success": True,
        "attempts": 1,
    },
    {
        "failure_class": "syntax_error",
        "traceback": seed_of("syntax_error"),
        "diff": "--- a/agent.py\n+++ b/agent.py\n@@\n-def heal(:\n+def heal(source, test, propose_diff):\n",
        "success": True,
        "attempts": 1,
    },
    {
        "failure_class": "runtime_error",
        "traceback": seed_of("runtime_error"),
        "diff": "--- a/workspace.py\n+++ b/workspace.py\n@@\n-    return (root / rel).resolve()\n+    resolved = (root / rel).resolve()\n+    if not str(resolved).startswith(str(root)):\n+        raise PathEscapeError(rel)\n+    return resolved\n",
        "success": False,
        "attempts": 4,
    },
]


# --------------------------------------------------------------------------
# Chained-exception corpus: the part where the keyword ladder cannot win
# --------------------------------------------------------------------------
# A transcript with two exception blocks has one *outermost* exception (the one
# printed last) and one incidental cause. The label is always the outermost
# class, so correctness depends on ordering, not on the presence of a keyword --
# which is precisely what a first-match rule ladder cannot represent.

_HEADS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "assertion_failure": (
        ("    assert allocate(10, 3) == [4, 3, 3]", "AssertionError: even split not implemented"),
        ("    assert ceil_cents(1.005) == 2", "AssertionError: half-up rounding"),
    ),
    "import_error": (
        ("    import cachetools", "ModuleNotFoundError: No module named 'cachetools'"),
        ("    from .legacy import bridge", "ImportError: attempted relative import with no known parent package"),
    ),
    "indentation_error": (
        ("    def apply_patch(", "IndentationError: expected an indented block after function definition"),
        ("        return x + 1", "IndentationError: unexpected indent"),
    ),
    "name_error": (
        ("    return normalise(rows)", "NameError: name 'normalise' is not defined"),
        ("    audit.log(entry)", "NameError: name 'audit' is not defined"),
    ),
    "syntax_error": (
        ("def heal(:", "SyntaxError: invalid syntax"),
        ("    return = value", "SyntaxError: invalid syntax"),
    ),
    "timeout": (
        ("    await asyncio.sleep(3600)", "TimeoutError: coroutine did not yield within budget"),
        ("    sock.recv(4096)", "TimeoutError: test run produced no output for 30s"),
    ),
    "type_error": (
        ("    total = amount + '1'", "TypeError: unsupported operand type(s) for +: 'int' and 'str'"),
        ("    return len(None)", "TypeError: object of type 'NoneType' has no len()"),
    ),
    "runtime_error": (
        ("    raise RuntimeError('workspace escaped jail root')", "RuntimeError: workspace escaped jail root"),
        ("    conn.commit()", "RuntimeError: database is locked"),
    ),
}

_CONNECTORS: Final[tuple[str, ...]] = (
    "\nE   \nE   The above exception was the direct cause of the following exception:\nE   \n",
    "\nE   \nE   During handling of the above exception, another exception occurred:\nE   \n",
)


def _chained(outer: tuple[str, str], inner: tuple[str, str], connector: str) -> str:
    (oframe, omsg), (iframe, imsg) = outer, inner
    return (
        "================ test session starts =================\n"
        f"{iframe}\nE   {imsg}\n"
        f"{connector}"
        f"{oframe}\nE   {omsg}\n"
        "1 failed in 0.42s\n"
    )


def _chained_cases() -> list[tuple[str, str]]:
    """Every ordered (inner, outer) pair of classes -- label is the outer class."""
    rows: list[tuple[str, str]] = []
    names = sorted(_HEADS)
    for outer in names:
        for inner in names:
            if inner == outer:
                continue
            for ci, connector in enumerate(_CONNECTORS):
                rows.append(
                    (
                        _chained(
                            _HEADS[outer][(ci + len(rows)) % len(_HEADS[outer])],
                            _HEADS[inner][len(rows) % len(_HEADS[inner])],
                            connector,
                        ),
                        outer,
                    )
                )
    return rows


HARD_TRACEBACKS: Final[list[tuple[str, str]]] = _chained_cases()

ALL_TRACEBACKS: Final[list[tuple[str, str]]] = SEED_TRACEBACKS + HARD_TRACEBACKS

_EXC_RE: Final = __import__("re").compile(r"\b([A-Z][A-Za-z]*(?:Error|Exception|Failure|Timeout))\b")


def is_hard(text: str) -> bool:
    """More than one exception class appears, so ordering decides the label."""
    return len(set(_EXC_RE.findall(text))) > 1


def hard_share(rows: list[tuple[str, str]] | None = None) -> float:
    """Fraction of a corpus whose label cannot be read off a keyword ladder."""
    rows = ALL_TRACEBACKS if rows is None else rows
    return round(sum(is_hard(text) for text, _ in rows) / max(len(rows), 1), 4)
