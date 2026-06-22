#!/usr/bin/env python3
"""
Backend Benchmark Demo

Thin wrapper over the first-class :mod:`mcframework.benchmark` subsystem. It times
the same Monte Carlo workload across every available execution backend
(sequential, thread, process, Torch CPU, and Apple-Silicon MPS when present) and
renders a four-panel performance figure.

The heavy lifting (measurement, summary, plotting) lives in the library, so this
file is intentionally tiny -- mirroring ``demo_convergence_gallery.py``.

Usage
-----
  python demo_backend_benchmark.py                      # full sweep, interactive
  MPLBACKEND=Agg python demo_backend_benchmark.py --quick --save backend_benchmark.png

This is exactly the ``mcframework-benchmark`` console script; run that directly
for the same result. Headless-safe with ``MPLBACKEND=Agg`` (``plt.show()`` is a
no-op). Keep ``matplotlib<3.11`` per AGENTS.md.
"""

from __future__ import annotations

import sys

from mcframework.benchmark import main

if __name__ == "__main__":
    sys.exit(main())
