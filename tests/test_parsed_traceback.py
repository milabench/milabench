import pytest

from milabench.validation.error import ParsedTraceback


class TestRaisedException:
    def test_returns_line_after_raise(self):
        lines = [
            "Traceback (most recent call last):",
            '  File "foo.py", line 10, in bar',
            "    raise ValueError(",
            "ValueError: something went wrong",
        ]
        tb = ParsedTraceback(lines=lines)
        assert tb.raised_exception() == "ValueError: something went wrong"

    def test_raise_on_last_line_does_not_overflow(self):
        """Regression: raised_idx + 1 == len(lines) must not IndexError."""
        lines = [
            "Traceback (most recent call last):",
            '  File "foo.py", line 5, in baz',
            "    raise RuntimeError('oops')",
        ]
        tb = ParsedTraceback(lines=lines)
        assert tb.raised_exception() == "    raise RuntimeError('oops')"

    def test_no_raise_returns_last_line(self):
        lines = [
            "Traceback (most recent call last):",
            '  File "foo.py", line 1, in <module>',
            "IndexError: list index out of range",
        ]
        tb = ParsedTraceback(lines=lines)
        assert tb.raised_exception() == "IndexError: list index out of range"

    def test_single_line_no_raise(self):
        lines = ["SomeError: unexpected"]
        tb = ParsedTraceback(lines=lines)
        assert tb.raised_exception() == "SomeError: unexpected"

    def test_single_line_with_raise(self):
        lines = ["    raise KeyError('x')"]
        tb = ParsedTraceback(lines=lines)
        assert tb.raised_exception() == "    raise KeyError('x')"


class TestFindRaise:
    def test_finds_raise_index_and_exception_name(self):
        lines = [
            '  File "foo.py", line 10',
            "    raise ValueError(",
            "ValueError: bad value",
        ]
        tb = ParsedTraceback(lines=lines)
        idx, name = tb.find_raise()
        assert idx == 1
        assert name == "ValueError"

    def test_returns_none_when_no_raise(self):
        lines = [
            "Traceback (most recent call last):",
            "TypeError: unsupported operand",
        ]
        tb = ParsedTraceback(lines=lines)
        idx, name = tb.find_raise()
        assert idx is None
        assert name is None


class TestAppendLine:
    def test_appends_normal_line(self):
        tb = ParsedTraceback(lines=["first line"])
        tb.append_line("second line\n")
        assert tb.lines == ["first line", "second line"]

    def test_merges_caret_lines(self):
        tb = ParsedTraceback(lines=["    ^^^"])
        tb.append_line("       ^^")
        assert tb.lines == ["    ^^^       ^^"]
