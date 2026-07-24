#!/usr/bin/env python3
"""Parse measure-log.<arch>.txt into results/<arch>.csv.

Usage: parse_measurements.py <aarch64|x86_64> [logpath]

One row per (kernel, variant, round). Every binary runs its own warmup +
internal min-of-N, so each round already carries one number per metric; the
five rounds are kept raw here and reduced downstream.
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if len(sys.argv) < 2 or sys.argv[1] not in ("aarch64", "x86_64"):
    sys.exit("usage: parse_measurements.py <aarch64|x86_64> [logpath]")
ARCH = sys.argv[1]
LOG = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "rustlane-bench" / f"measure-log.{ARCH}.txt"

MS_LINE = re.compile(r"^MS[_ ]\s*(?:([A-Za-z_]\S*)\s+)?([0-9.]+)\s*$")
MANDEL_SPMD = re.compile(r"^mandelbrot\((direct|export),N=\d+\):\s+([0-9.]+) ms")

# kernel -> (ispc stem, metric name printed by the ISPC/C++ driver, metric printed by the rustlane bin)
KERNELS = [
    ("mandelbrot",    "mandel",  "mandelbrot_ispc",    "mandelbrot_serial",    "mandelbrot", "export"),
    ("black_scholes", "options", "black_scholes_ispc", "black_scholes_serial", "options",    "black_scholes"),
    ("binomial_put",  "options", "binomial_put_ispc",  "binomial_put_serial",  "options",    "binomial_put"),
    ("stencil",       "stencil", "stencil_ispc",       "stencil_serial",       "stencil",    "stencil"),
    ("volume",        "volume",  "ispc",               "serial",               "volume",     "main"),
    ("ao",            "ao",      "ispc",               "serial",               "ao",         "ao"),
    ("rt",            "rt",      "ispc",               "serial",               "rt",         "main"),
]

TARGETS = {
    "aarch64": {"ispc_narrow": "neon-i32x4", "ispc_wide": "neon-i32x8"},
    "x86_64":  {"ispc_narrow": "avx2-i32x8", "ispc_wide": "avx512skx-i32x16"},
}[ARCH]

runs = defaultdict(lambda: defaultdict(list))   # binpath -> metric -> [ms per round]
ok = defaultdict(int)
failed = defaultdict(int)
exits = []
cur = None
for line in LOG.read_text().splitlines():
    if line.startswith("### round="):
        cur = line.split("bin=")[1]
        continue
    if line.startswith("### exit="):
        exits.append(int(line.split("=")[1]))
        continue
    if cur is None:
        continue
    m = MS_LINE.match(line)
    if m:
        runs[cur][(m.group(1) or "main").lower()].append(float(m.group(2)))
    m = MANDEL_SPMD.match(line)
    if m:
        runs[cur][m.group(1)].append(float(m.group(2)))
    if re.search(r": OK\b", line):
        ok[cur] += 1
    if re.search(r"VALIDATION FAILED|: FAIL\b", line):
        failed[cur] += 1

assert all(e == 0 for e in exits), f"nonzero exits: {exits}"


def series(binname, metric):
    v = runs[binname][metric]
    assert len(v) == 5, f"{binname}/{metric}: got {len(v)} values: {v}"
    return v


rows = []
for kernel, stem, ispc_metric, cpp_metric, rlbin, rl_metric in KERNELS:
    a, b = f"bench_{stem}_a", f"bench_{stem}_b"
    for variant, target, values in [
        ("scalar",      "clang++ -O3 -fno-vectorize", series(b, cpp_metric)),
        ("cpp_autovec", "clang++ -O3",                series(a, cpp_metric)),
        ("ispc_narrow", TARGETS["ispc_narrow"],       series(a, ispc_metric)),
        ("ispc_wide",   TARGETS["ispc_wide"],         series(b, ispc_metric)),
        ("rustlane",    "runtime dispatch",           series(f"target/release/{rlbin}", rl_metric)),
    ]:
        for rnd, ms in enumerate(values, start=1):
            rows.append({"arch": ARCH, "kernel": kernel, "variant": variant,
                         "target": target, "round": rnd, "ms": ms})

out = ROOT / "results" / f"{ARCH}.csv"
out.parent.mkdir(exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["arch", "kernel", "variant", "target", "round", "ms"])
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out}  ({len(rows)} rows)")

anyfail = sum(failed.values())
if anyfail:
    print(f"!! {anyfail} validation-failed rounds: {[k for k, v in failed.items() if v]}")
else:
    print(f"validation OK in every round ({sum(ok.values())} OK lines)")
