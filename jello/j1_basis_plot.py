#!/usr/bin/env python3
"""
Plot cubic basis functions on the unit interval [0, 1].

Basis (t in [0,1], s := 1-t):
  v_0^0(t) := s^3 + 3 s^2 t
  v_0^1(t) := 3 s t^2 + t^3
  v_1^0(t) := s^2 t
  v_1^1(t) := - s t^2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PlotConfig:
    num_points: int = 2001
    linewidth: float = 2.2
    dpi: int = 300
    xlim: tuple[float, float] = (0.0, 1.0)


def basis_functions(t: np.ndarray) -> dict[str, np.ndarray]:
    s = 1.0 - t
    v00 = s**3 + 3.0 * s**2 * t
    v10 = s**2 * t
    v01 = 3.0 * s * t**2 + t**3
    v11 = -s * t**2
    return {
        r"$v_{0}^{0}(t)$": v00,
        r"$v_{1}^{0}(t)$": v10,
        r"$v_{0}^{1}(t)$": v01,
        r"$v_{1}^{1}(t)$": v11,
    }


def make_plot(*, out_base: Path, show: bool, cfg: PlotConfig) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Reasonable, journal-ish defaults without requiring LaTeX.
    mpl.rcParams.update(
        {
            "figure.dpi": cfg.dpi,
            "savefig.dpi": cfg.dpi,
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 1.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "lines.linewidth": cfg.linewidth,
        }
    )

    t = np.linspace(cfg.xlim[0], cfg.xlim[1], cfg.num_points, dtype=float)
    values = basis_functions(t)

    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    for label, v in values.items():
        ax.plot(t, v, label=label)

    ax.set_xlim(*cfg.xlim)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(loc="best", frameon=False, ncols=2)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")

    if show:
        plt.show()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Make a single plot of the cubic reference-element basis functions."
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("basis_functions"),
        help="Output file base path (no extension). Writes .pdf and .png. Default: %(default)s",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Display interactively (also still writes output files).",
    )
    p.add_argument(
        "--num-points",
        type=int,
        default=PlotConfig.num_points,
        help="Number of sample points in t. Default: %(default)s",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = PlotConfig(num_points=int(args.num_points))
    make_plot(out_base=args.out, show=bool(args.show), cfg=cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

