"""Unit tests for hyphaeon._progress.ChunkProgress.

CPU-only, no model, no GPU. Fully deterministic: the wall clock is controlled
via monkeypatching ``hyphaeon._progress.time.time`` and TTY-ness is controlled
by forcing ``ChunkProgress.is_tty`` (never any real sleeping).
"""
import re

import pytest

import hyphaeon._progress as _progress
from hyphaeon._progress import ChunkProgress


@pytest.fixture
def fake_clock(monkeypatch):
    """A controllable clock for hyphaeon._progress.time.time().

    Returns a small handle whose ``.t`` attribute is the current fake time;
    set it to advance the clock deterministically. No real time passes.
    """
    class Clock:
        t = 1000.0

    clock = Clock()
    monkeypatch.setattr(_progress.time, "time", lambda: clock.t)
    return clock


def _non_tty(total, label="Predict", unit="codon", enabled=True):
    """Construct a ChunkProgress forced to the non-TTY (fresh-line) path."""
    p = ChunkProgress(total, label, unit=unit, enabled=enabled)
    p.is_tty = False
    return p


class TestDisabled:
    def test_update_and_finish_are_noops(self, capsys):
        """enabled=False -> update() and finish() print nothing at all."""
        p = ChunkProgress(100, "Predict", enabled=False)
        p.is_tty = False
        p.update(0)
        p.update(50)
        p.update(100)
        p.finish()
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


class TestNonTty:
    def test_update_prints_fresh_line(self, capsys, fake_clock):
        """Non-TTY update() prints a fresh flushed line with label/unit/done/total/pct."""
        p = _non_tty(200, label="Disease", unit="codon")
        fake_clock.t += 2.0
        p.update(40)
        out = capsys.readouterr().out
        # A fresh line ending in newline (print), no carriage return.
        assert "\r" not in out
        assert out.endswith("\n")
        assert "[Disease]" in out
        assert "codon" in out
        assert "40/" in out
        assert "200" in out
        assert "%" in out

    def test_finish_prints_no_extra_newline(self, capsys, fake_clock):
        """On non-TTY, finish() must NOT emit an extra trailing newline."""
        p = _non_tty(10)
        p.finish()
        assert capsys.readouterr().out == ""


class TestCompletionBypassesThrottle:
    def test_completion_always_emits(self, capsys, fake_clock):
        """update(total) prints the 100.0% line even immediately after another update."""
        p = _non_tty(100)
        p.update(10)             # first print, sets _last_update
        capsys.readouterr()      # drain
        # No clock advance: still well within the 0.5s throttle window.
        p.update(100)            # done >= total must bypass the throttle
        out = capsys.readouterr().out
        assert out != ""
        assert "100.0%" in out
        assert "100/  100" in out


class TestThrottle:
    def test_rapid_updates_below_total_throttled(self, capsys, fake_clock):
        """Two rapid update(x<total) within <0.5s -> only the first prints."""
        p = _non_tty(1000)
        fake_clock.t += 1.0
        p.update(100)
        first = capsys.readouterr().out
        assert "100/" in first
        # Advance less than the 0.5s throttle threshold.
        fake_clock.t += 0.1
        p.update(200)
        second = capsys.readouterr().out
        assert second == ""

    def test_update_emits_again_after_throttle_window(self, capsys, fake_clock):
        """After >=0.5s elapses, a sub-total update prints again."""
        p = _non_tty(1000)
        fake_clock.t += 1.0
        p.update(100)
        capsys.readouterr()
        fake_clock.t += 0.6  # past the 0.5s throttle
        p.update(200)
        assert "200/" in capsys.readouterr().out


class TestMathSafety:
    def test_total_zero_update_zero_no_zerodivision(self, capsys, fake_clock):
        """total=0 and done=0 must not raise ZeroDivisionError."""
        p = _non_tty(0)
        # elapsed is 0.0 (clock unchanged) — guards must hold on both axes.
        p.update(0)
        out = capsys.readouterr().out
        assert "0/    0" in out
        assert "%" in out

    def test_total_zero_with_elapsed_no_zerodivision(self, capsys, fake_clock):
        """total=0 with nonzero elapsed still safe."""
        p = _non_tty(0)
        fake_clock.t += 3.0
        p.update(0)
        assert capsys.readouterr().out != ""


class TestContentFormat:
    def test_line_shape(self, capsys, fake_clock):
        """The printed line matches the expected shape (label, x/total, %, rate, Elapsed, ETA)."""
        p = _non_tty(500, label="Attribution", unit="codon")
        fake_clock.t += 10.0  # deterministic elapsed
        p.update(250)
        line = capsys.readouterr().out.strip()
        # [Attribution] codon   250/  500 ( 50.0%) |   25.0 codon/s | Elapsed:  10.0s | ETA:  10.0s
        pattern = (
            r"^\[Attribution\] codon\s+250/\s*500 \(\s*50\.0%\) \| "
            r"\s*[\d.]+ codon/s \| Elapsed:\s*10\.0s \| ETA:\s*[\d.]+s$"
        )
        assert re.match(pattern, line), f"line did not match expected shape: {line!r}"
        # Spot-check the individual pieces the format contract promises.
        assert "[Attribution]" in line
        assert "250/" in line
        assert "50.0%" in line
        assert "Elapsed:" in line
        assert "ETA:" in line
        # One decimal place on the percentage.
        assert re.search(r"\d+\.\d%", line)


class TestCoverageGuard:
    """Light guard that the shared bar stays wired into every model-running command."""

    @pytest.mark.parametrize("module", ["cli", "disease", "epistasis", "attribution"])
    def test_chunkprogress_referenced(self, module):
        """ChunkProgress must remain imported/used in each command module.

        Guards against a future edit silently dropping the live progress bars.
        Uses source inspection so no model / GPU is required.
        """
        import importlib
        import inspect

        mod = importlib.import_module(f"hyphaeon.{module}")
        src = inspect.getsource(mod)
        assert "ChunkProgress" in src, (
            f"hyphaeon.{module} no longer references ChunkProgress"
        )
