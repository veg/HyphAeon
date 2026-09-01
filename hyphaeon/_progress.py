"""
Shared live progress bar for model-running subcommands.

A single reusable carriage-return progress bar lifted faithfully from the
filter baseline/re-eval bars (filter.py) and the DMS bar (epistasis.py). Every
per-chunk inference loop that just needs the plain "codon x/n, %, rate,
elapsed, ETA" bar should construct a ``ChunkProgress`` and call ``.update()``
per chunk and ``.finish()`` at the end.

The richer DMS-specific bar (adds mut/s + running mean plasticity Phi) stays
in ``run_insilico_selection_dms`` as-is; this generic helper covers the plain
inference loops (predict, busted single-file, disease, phenotype/epistasis
attribution).

TTY vs non-TTY behaviour mirrors the DMS bar:
  * on a TTY it rewrites a single line via carriage-return + clear-to-EOL;
  * otherwise it prints a fresh (flushed) line so progress is visible under
    SLURM logs / pipes.
"""

import sys
import time


class ChunkProgress:
    """Reusable per-chunk live progress bar.

    Parameters
    ----------
    total : int
        Total number of units to process (e.g. number of variable codons).
    label : str
        Bracketed label shown at the start of the bar, e.g. ``"Predict"``.
    unit : str
        Unit name printed in the bar (default ``"codon"``).
    enabled : bool
        When False, all methods are no-ops (mirrors the ``progress`` flag).
    """

    def __init__(self, total, label, unit="codon", enabled=True):
        self.total = int(total)
        self.label = label
        self.unit = unit
        self.enabled = bool(enabled)
        self.t_start = time.time()
        self.is_tty = sys.stdout.isatty()

    def update(self, done):
        """Report ``done`` units completed; throttled to >=0.5s or completion."""
        if not self.enabled:
            return
        now = time.time()
        # Throttle refreshes to avoid flooding logs; always emit on completion.
        last = getattr(self, "_last_update", None)
        if last is not None and done < self.total and (now - last) < 0.5:
            return
        self._last_update = now

        done = int(done)
        pct = (done / max(1, self.total)) * 100.0
        elapsed = now - self.t_start
        rate = done / max(1e-3, elapsed)
        eta = (self.total - done) / max(1e-3, rate)
        line = (
            f"[{self.label}] {self.unit} {done:5d}/{self.total:5d} "
            f"({pct:5.1f}%) | {rate:6.1f} {self.unit}/s | "
            f"Elapsed: {elapsed:5.1f}s | ETA: {eta:5.1f}s"
        )
        if self.is_tty:
            sys.stdout.write("\r\033[K" + line)
            sys.stdout.flush()
        else:
            print(line, flush=True)

    def finish(self):
        """Emit the trailing newline on a TTY so the bar line is not overwritten."""
        if not self.enabled:
            return
        if self.is_tty:
            sys.stdout.write("\n")
            sys.stdout.flush()
