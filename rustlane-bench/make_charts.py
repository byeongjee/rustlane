#!/usr/bin/env python3
"""Draw docs/bench-<arch>.png from results/<arch>.csv.

Usage: make_charts.py <aarch64|x86_64> ...

Each bar is the mean speedup over the scalar C++ baseline across the measurement
rounds; the whisker is a 95% t confidence interval on that mean. Speedup is
formed per round (round i of the baseline against round i of the variant), so
the interval carries the run-to-run spread of both.
"""
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots 1-3 of the validated palette, plus a neutral for the baseline.
RUSTLANE = "#2a78d6"
ISPC_MATCHED_WIDTH = "#eb6834"
ISPC_OTHER_WIDTH = "#1baf7a"
CPP_AUTOVEC = "#898781"

# Two-sided 95% t multipliers by sample count.
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}

TITLES = {
    "aarch64": "Apple M2 Pro — aarch64 / NEON",
    "x86_64": "AMD Ryzen 9 7900X (Zen4) — x86-64",
}

ENV = {
    "aarch64": "MacBook Pro (Mac14,10), Apple M2 Pro 8P+4E, 32 GB, macOS 26.5.2  ·  "
               "rustc 1.92.0-nightly (844264add 2025-10-14)  ·  ISPC 1.30.0 (LLVM 22.1.0)  ·  "
               "clang++ 22.1.8 (Homebrew), -O3",
    "x86_64": "AMD Ryzen 9 7900X (Zen4) 12C/24T, 45 GB, Ubuntu 7.0.0-22-generic, single-tenant  ·  "
              "rustc 1.92.0-nightly (4b94758d2 2025-10-13)  ·  ISPC 1.31.0 (LLVM 23.0.0)  ·  "
              "clang++-22 22.1.2, -O3",
}

KERNEL_ORDER = ["mandelbrot", "black_scholes", "binomial_put", "stencil", "volume", "ao", "rt"]
VARIANT_ORDER = ["cpp_autovec", "ispc_narrow", "ispc_wide", "rustlane"]


def per_round_speedup(arch):
    """Long-format speedup over the scalar baseline, paired within each round."""
    df = pd.read_csv(ROOT / "results" / f"{arch}.csv")
    base = df[df.variant == "scalar"].set_index(["kernel", "round"]).ms
    df = df[df.variant != "scalar"].copy()
    df["speedup"] = [base[(k, r)] / ms for k, r, ms in zip(df.kernel, df["round"], df.ms)]
    return df.drop(columns="ms")


def with_geomean(df):
    """Append a 'geomean' kernel: the geometric mean across kernels, computed per round."""
    g = (
        df.groupby(["variant", "target", "round"])
        .speedup.apply(lambda s: math.exp(sum(map(math.log, s)) / len(s)))
        .reset_index()
    )
    g["kernel"] = "geomean"
    return pd.concat([df, g], ignore_index=True)


def ci95(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return T95.get(n, 1.96) * sd / math.sqrt(n)


def legend_label(variant, target):
    if variant == "cpp_autovec":
        return "C++ auto-vec"
    if variant == "rustlane":
        return "rustlane"
    return f"ISPC {target}"


def series_color(variant, target, rustlane_lanes):
    if variant == "rustlane":
        return RUSTLANE
    if variant == "cpp_autovec":
        return CPP_AUTOVEC
    return ISPC_MATCHED_WIDTH if f"x{rustlane_lanes}" in target else ISPC_OTHER_WIDTH


def draw(arch):
    df = with_geomean(per_round_speedup(arch))
    # rustlane dispatches to the widest available target; on x86 that is the 16-wide one.
    rustlane_lanes = 16 if any("avx512" in t for t in df.target) else 8

    variants = df[["variant", "target"]].drop_duplicates().set_index("variant").loc[VARIANT_ORDER]
    hue_order = [legend_label(v, t) for v, t in zip(variants.index, variants.target)]
    colors = {
        legend_label(v, t): series_color(v, t, rustlane_lanes)
        for v, t in zip(variants.index, variants.target)
    }
    df["series"] = [legend_label(v, t) for v, t in zip(df.variant, df.target)]

    stats = df.groupby(["kernel", "series"]).speedup.agg(mean="mean", ci=ci95)
    top = (stats["mean"] + stats.ci).max()
    order = KERNEL_ORDER + ["geomean"]

    sns.set_theme(style="white", font="Helvetica Neue", rc={"figure.facecolor": SURFACE})
    fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=200)
    ax.set_facecolor(SURFACE)

    sns.barplot(
        df, x="kernel", y="speedup", hue="series", order=order, hue_order=hue_order,
        palette=colors, estimator="mean", errorbar=None,
        ax=ax, width=0.8, saturation=1.0,
    )

    for container, series in zip(ax.containers, hue_order):
        for bar, kernel in zip(container, order):
            mean, ci = stats.loc[(kernel, series)]
            x = bar.get_x() + bar.get_width() / 2
            ax.errorbar(x, mean, yerr=ci, fmt="none", ecolor=INK_2,
                        elinewidth=0.9, capsize=1.8, capthick=0.9)
            text = f"{mean:.2f}" if mean < 10 else f"{mean:.1f}"
            ax.text(x, mean + ci + top * 0.012, text,
                    ha="center", va="bottom", fontsize=5.8, color=INK_2)

    ax.axhline(1.0, color=AXIS, lw=1, ls=(0, (4, 3)), zorder=0)
    ax.axvline(6.5, color=AXIS, lw=0.8, zorder=0)
    ax.set_ylim(0, top * 1.16)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8, colors=MUTED, length=0)
    for tick in ax.get_xticklabels():
        tick.set_color(INK_2)
        if tick.get_text() == "geomean":
            tick.set_fontweight("bold")
            tick.set_color(INK)
    sns.despine(ax=ax, left=True, bottom=False)
    ax.spines["bottom"].set_color(AXIS)

    ax.legend(
        loc="lower left", bbox_to_anchor=(-0.048, 1.005), ncol=4, frameon=False,
        fontsize=8.5, handlelength=1.1, handleheight=1.1, columnspacing=1.6, labelcolor=INK_2,
    )
    fig.suptitle(TITLES[arch], x=0.008, y=0.965, ha="left", fontsize=11.5, color=INK, fontweight="bold")
    fig.text(0.008, 0.885,
             f"Speedup over scalar C++ — higher is better.  rustlane runs {rustlane_lanes} lanes.  "
             f"Whisker: 95% CI over {df['round'].nunique()} rounds.",
             ha="left", fontsize=8.5, color=INK_2)
    fig.text(0.008, 0.022, ENV[arch], ha="left", fontsize=6.4, color=MUTED)

    fig.subplots_adjust(left=0.048, right=0.995, top=0.80, bottom=0.15)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"bench-{arch}.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


if len(sys.argv) < 2:
    sys.exit("usage: make_charts.py <aarch64|x86_64> ...")
for arch in sys.argv[1:]:
    draw(arch)
