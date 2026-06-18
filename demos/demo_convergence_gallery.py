#!/usr/bin/env python3
"""
Convergence Gallery Demo

The headline "show off the plumbing" artifact: every oracle-backed simulation is swept
over increasing sample sizes and plotted converging to its *known* answer. If the
Monte Carlo machinery is correct, the absolute error decays like 1/sqrt(n) and the
estimate stays inside its confidence interval. This is correctness you can see.

For each built-in simulation that declares an
:meth:`~mcframework.core.MonteCarloSimulation.analytic_reference` (Pi, Portfolio,
Black-Scholes path, European Black-Scholes), it produces:

  (a) a matplotlib log-log error-vs-n plot with a 1/sqrt(n) reference line, and
  (b) a printed Markdown convergence table.

Usage
-----
  python demo_convergence_gallery.py            # full sweep, interactive
  MPLBACKEND=Agg python demo_convergence_gallery.py --quick --save gallery.png

Headless-safe: with ``MPLBACKEND=Agg`` the ``plt.show()`` is a no-op. Keep
``matplotlib<3.11`` per AGENTS.md.
"""

from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from mcframework import (
    BlackScholesPathSimulation,
    BlackScholesSimulation,
    ConvergenceReport,
    PiEstimationSimulation,
    PortfolioSimulation,
    validate_convergence,
)

# (label, simulation factory, run params, sample sizes to sweep)
GALLERY: list[tuple[str, type, dict, tuple[int, ...]]] = [
    ("Pi Estimation", PiEstimationSimulation, {}, (1_000, 4_000, 16_000, 64_000, 256_000)),
    ("Portfolio (GBM)", PortfolioSimulation, {}, (1_000, 4_000, 16_000, 64_000, 256_000)),
    ("Black-Scholes Path", BlackScholesPathSimulation, {}, (1_000, 4_000, 16_000, 64_000, 256_000)),
    ("BS European Call", BlackScholesSimulation, {"option_type": "call"},
     (1_000, 4_000, 16_000, 64_000, 256_000)),
]

# Quick mode trims the sweep so the demo (and its smoke test) run in well under a second.
_QUICK_SIZES = (500, 2_000, 8_000)


def sweep_convergence(
    factory: type,
    sizes: tuple[int, ...],
    *,
    seed: int = 0,
    **params,
) -> list[ConvergenceReport]:
    """Run :func:`validate_convergence` for one simulation across increasing ``n``."""
    return [validate_convergence(factory(), n, seed=seed, **params) for n in sizes]


def build_markdown_table(label: str, reports: list[ConvergenceReport]) -> str:
    """Render a Markdown convergence table for one simulation's sweep."""
    source = reports[0].reference_source if reports else ""
    lines = [
        f"### {label}",
        f"*Oracle source: {source}*",
        "",
        "| n | estimate | oracle | abs error | abs error / SE | status |",
        "| ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for r in reports:
        err_in_se = (r.abs_error / r.se) if (r.abs_error is not None and r.se) else float("nan")
        lines.append(
            f"| {r.n:,} | {r.estimate:.6g} | {r.oracle:.6g} | "
            f"{r.abs_error:.3g} | {err_in_se:.2f} | {r.status} |"
        )
    return "\n".join(lines)


def run_gallery(*, quick: bool = False, seed: int = 0) -> dict[str, list[ConvergenceReport]]:
    """Sweep every oracle-backed simulation and return its reports keyed by label."""
    gallery: dict[str, list[ConvergenceReport]] = {}
    for label, factory, params, sizes in GALLERY:
        sweep_sizes = _QUICK_SIZES if quick else sizes
        gallery[label] = sweep_convergence(factory, sweep_sizes, seed=seed, **params)
    return gallery


def create_convergence_plot(gallery: dict[str, list[ConvergenceReport]]):
    """Build a log-log error-vs-n figure with a 1/sqrt(n) reference per simulation."""
    n_panels = len(gallery)
    ncols = 2
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.2 * nrows), squeeze=False)
    flat_axes = axes.ravel()

    for ax, (label, reports) in zip(flat_axes, gallery.items(), strict=False):
        ns = np.array([r.n for r in reports], dtype=float)
        abs_err = np.array([r.abs_error for r in reports], dtype=float)
        se = np.array([r.se for r in reports], dtype=float)

        ax.loglog(ns, abs_err, "o-", label="abs error", color="#1f77b4")
        # 1/sqrt(n) guide anchored at the first point: the theoretical MC decay rate.
        guide = abs_err[0] * np.sqrt(ns[0]) / np.sqrt(ns)
        ax.loglog(ns, guide, "--", color="#888888", label=r"$1/\sqrt{n}$ reference")
        # Shade the pass region (within sigma_tol * SE): staying below the band is "pass".
        sigma_tol = reports[0].sigma_tol
        ax.fill_between(ns, 0, sigma_tol * se, color="#2ca02c", alpha=0.15,
                        label=f"within {sigma_tol:g}·SE")
        ax.set_title(label)
        ax.set_xlabel("n (samples)")
        ax.set_ylabel("|estimate − oracle|")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=8)

    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    fig.suptitle("Monte Carlo convergence to known answers (oracles)", fontsize=13)
    fig.tight_layout()
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the convergence gallery."""
    parser = argparse.ArgumentParser(description="Monte Carlo convergence gallery")
    parser.add_argument("--quick", action="store_true",
                        help="Use a trimmed sample-size sweep for a fast run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default 0).")
    parser.add_argument("--save", metavar="PATH", default=None,
                        help="Save the figure to PATH instead of prompting.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the sweep, print Markdown tables, and render the convergence figure."""
    args = parse_args(argv)
    gallery = run_gallery(quick=args.quick, seed=args.seed)

    print("# Convergence Gallery\n")
    all_pass = True
    for label, reports in gallery.items():
        print(build_markdown_table(label, reports))
        print()
        all_pass = all_pass and all(r.status == "pass" for r in reports)

    print("All simulations converged to their oracles."
          if all_pass else "WARNING: at least one simulation failed to converge.")

    fig = create_convergence_plot(gallery)

    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"\nSaved figure to {args.save}")
    else:
        plt.show()
        try:
            if input("\nSave gallery figure to file? (y/N): ").lower().strip() == "y":
                fig.savefig("convergence_gallery.png", dpi=200, bbox_inches="tight")
                print("Saved to convergence_gallery.png")
        except EOFError:
            pass

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
