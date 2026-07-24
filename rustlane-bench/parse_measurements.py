#!/usr/bin/env python3
"""Parse measure-log.<arch>.txt into RESULTS.<arch>.json under the
scalar / cpp_autovec / ispc_narrow / ispc_wide / rustlane taxonomy.

Usage: parse_measurements.py <aarch64|x86_64> [logpath]
Each binary reports its own warmup + internal min-of-N; we take the min across
the 5 rounds per (binary, metric).
"""
import json
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
MANDEL_SPMD = re.compile(r"^mandelbrot\((direct|export),N=8\):\s+([0-9.]+) ms")

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

def mn(binname, metric):
    v = runs[binname][metric]
    assert len(v) == 5, f"{binname}/{metric}: got {len(v)} values: {v}"
    return min(v)

# bench, stem, ispc_label, serial_label, rustlane_bin, rustlane_metric, validation
KERNELS = [
    ("mandelbrot",    "mandel",  "mandelbrot_ispc",    "mandelbrot_serial",    "mandelbrot", "direct",        "self-checksum; fma boundary pixels may differ per arch"),
    ("black_scholes", "options", "black_scholes_ispc", "black_scholes_serial", "options",    "black_scholes", "vs ISPC ref file"),
    ("binomial_put",  "options", "binomial_put_ispc",  "binomial_put_serial",  "options",    "binomial_put",  "vs ISPC ref file"),
    ("stencil",       "stencil", "stencil_ispc",       "stencil_serial",       "stencil",    "stencil",       "vs ISPC ref file (fma tolerance)"),
    ("volume",        "volume",  "ispc",               "serial",               "volume",     "main",          "vs ISPC ref file at 1e-3"),
    ("ao",            "ao",      "ispc",               "serial",               "ao",         "ao",            "statistical (RNG streams differ by gang width)"),
    ("rt",            "rt",      "ispc",               "serial",               "rt",         "main",          "bit-exact vs serial ground truth"),
]

RL = {
    "aarch64": [("rustlane", "target/release")],
    "x86_64":  [("rustlane_v3", "target-v3/release"), ("rustlane_native", "target-native/release")],
}[ARCH]
TARGETS = {
    "aarch64": {"ispc_narrow": "neon-i32x4", "ispc_wide": "neon-i32x8"},
    "x86_64":  {"ispc_narrow": "avx2-i32x8", "ispc_wide": "avx512skx-i32x16"},
}[ARCH]

rows = []
for bench, stem, il, sl, rlbin, rlm, val in KERNELS:
    a, b = f"bench_{stem}_a", f"bench_{stem}_b"
    ms = {
        "scalar":      mn(b, sl),
        "cpp_autovec": mn(a, sl),
        "ispc_narrow": mn(a, il),
        "ispc_wide":   mn(b, il),
    }
    for vname, d in RL:
        ms[vname] = mn(f"{d}/{rlbin}", rlm)
    rl_best = min(ms[v] for v, _ in RL)
    derived = {
        "rustlane_best_over_ispc_narrow": round(rl_best / ms["ispc_narrow"], 4),
        "rustlane_best_over_ispc_wide":   round(rl_best / ms["ispc_wide"], 4),
        "speedup_scalar_to_autovec":      round(ms["scalar"] / ms["cpp_autovec"], 2),
        "speedup_scalar_to_ispc_wide":    round(ms["scalar"] / ms["ispc_wide"], 2),
        "speedup_scalar_to_rustlane_best":round(ms["scalar"] / rl_best, 2),
    }
    rows.append({
        "bench": bench, "kernel": bench, "arch": ARCH,
        "ms": {k: round(v, 4) for k, v in ms.items()},
        "targets": TARGETS, "derived": derived,
        "validated": True, "validation": val,
    })

geo = 1.0
for r in rows:
    geo *= r["derived"]["rustlane_best_over_ispc_narrow"]
geo = geo ** (1.0 / len(rows))

# validation status per rustlane bin (each should print OK 5x)
RLK = ["mandelbrot", "options", "stencil", "volume", "ao", "rt"]
val_status = {f"{d}/{k}": {"ok": ok.get(f"{d}/{k}", 0), "failed": failed.get(f"{d}/{k}", 0)}
              for _, d in RL for k in RLK}

out = {
    "_status": f"{ARCH} sweep, 5 interleaved rounds, min across rounds",
    "arch": ARCH,
    "geomean_rustlane_best_over_ispc_narrow": round(geo, 4),
    "rows": rows,
    "validation_rounds": val_status,
    "raw_rounds": {b: dict(m) for b, m in runs.items()},
}
(ROOT / "rustlane-bench" / f"RESULTS.{ARCH}.json").write_text(json.dumps(out, indent=2) + "\n")

hdr = f"{'kernel':16s} {'scalar':>10s} {'autovec':>10s} {'ispc_nrw':>10s} {'ispc_wide':>10s} " + " ".join(f"{v:>14s}" for v, _ in RL)
print(hdr)
for r in rows:
    m = r["ms"]
    tail = " ".join(f"{m[v]:14.3f}" for v, _ in RL)
    print(f"{r['kernel']:16s} {m['scalar']:10.3f} {m['cpp_autovec']:10.3f} {m['ispc_narrow']:10.3f} {m['ispc_wide']:10.3f} {tail}")
print(f"\ngeomean rustlane-best / ISPC-narrow = {geo:.4f}")
anyfail = sum(v['failed'] for v in val_status.values())
if anyfail:
    print(f"\n!! {anyfail} validation-failed rounds: {[k for k, v in val_status.items() if v['failed']]}")
