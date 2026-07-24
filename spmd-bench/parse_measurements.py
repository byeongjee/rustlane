#!/usr/bin/env python3
"""Parse spmd-bench/measure-log.txt into RESULTS.final.json.

Takes the minimum across rounds per (binary, metric); each binary already
reports its own warmup + internal min-of-N.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "spmd-bench" / "measure-log.txt"

MS_LINE = re.compile(r"^MS[_ ]\s*(?:([A-Za-z_]\S*)\s+)?([0-9.]+)\s*$")
MANDEL_SPMD = re.compile(r"^mandelbrot\((direct|export),N=8\):\s+([0-9.]+) ms")

runs = defaultdict(lambda: defaultdict(list))  # bin -> metric -> [ms per round]
ok_counts = defaultdict(int)
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
    if re.match(r"^\w+: OK", line):
        ok_counts[cur] += 1

assert all(e == 0 for e in exits), f"nonzero exits: {exits}"
for b in ("mandelbrot", "options", "stencil", "volume", "ao", "rt"):
    assert ok_counts[f"target/release/{b}"] == 5, f"{b}: {ok_counts}"

def mn(binname, metric):
    v = runs[binname][metric]
    assert len(v) == 5, f"{binname}/{metric}: {v}"
    return min(v)

def ispc_pair(stem, kernel, serial_key=None):
    serial_key = serial_key or kernel.replace("_ispc", "") + "_serial"
    x4 = mn(f"bench_{stem}_x4", kernel)
    x8 = mn(f"bench_{stem}_x8", kernel)
    serial = min(mn(f"bench_{stem}_x4", serial_key),
                 mn(f"bench_{stem}_x8", serial_key))
    return serial, x4, x8

rows = []
def add(bench, kernel, serial, x4, x8, spmd, validation):
    rows.append({
        "bench": bench, "kernel": kernel,
        "serial_ms": serial, "ispc_x4_ms": x4, "ispc_x8_ms": x8, "spmd_ms": spmd,
        "spmd_over_ispc_best": round(spmd / min(x4, x8), 4),
        "speedup_vs_serial_spmd": round(serial / spmd, 2),
        "speedup_vs_serial_ispc_best": round(serial / min(x4, x8), 2),
        "validated": True, "validation": validation,
    })

s, x4, x8 = ispc_pair("mandel", "mandelbrot_ispc")
add("mandelbrot", "mandelbrot", s, x4, x8, mn("target/release/mandelbrot", "direct"),
    "checksum 27304085 all rounds; export == direct")

s, x4, x8 = ispc_pair("options", "black_scholes_ispc")
add("options", "black_scholes", s, x4, x8, mn("target/release/options", "black_scholes"),
    "bit-exact vs ISPC reference (max_rel 0, sum 13102023)")

s, x4, x8 = ispc_pair("options", "binomial_put_ispc")
add("options", "binomial_put", s, x4, x8, mn("target/release/options", "binomial_put"),
    "bit-exact vs ISPC reference (max_rel 0, sum 12344328)")

s, x4, x8 = ispc_pair("stencil", "stencil_ispc")
add("stencil", "stencil", s, x4, x8, mn("target/release/stencil", "stencil"),
    "checksum_rel 1.09e-10, max_abs 4.4e-5 (fma tolerance; ref is fma-contracted)")

s, x4, x8 = ispc_pair("volume", "ispc", "serial")
add("volume", "volume", s, x4, x8, mn("target/release/volume", "main"),
    "checksum_rel 2.68e-8, 0/1060864 mismatches at 1e-3")

s, x4, x8 = ispc_pair("ao", "ispc", "serial")
add("ao", "ao", s, x4, x8, mn("target/release/ao", "ao"),
    "statistical: 0.0224% checksum rel err (RNG streams differ by gang width)")

s, x4, x8 = ispc_pair("rt", "ispc", "serial")
add("rt", "rt", s, x4, x8, mn("target/release/rt", "main"),
    "bit-exact vs serial ground truth (0/810000, checksum_rel 0)")

ratios = [r["spmd_over_ispc_best"] for r in rows]
geomean = 1.0
for r in ratios:
    geomean *= r
geomean = geomean ** (1.0 / len(ratios))

out = {
    "_status": "FINAL (5-round single-tenant sweep, min across rounds)",
    "machine": "MacBook Pro (Mac14,10), Apple M2 Pro, 12 cores (8P+4E), 32 GB, macOS 26.5.2",
    "toolchain": "rustc 1.92.0-nightly (2025-10-14); ISPC 1.30.0 (LLVM 22.1.0); clang++ (Homebrew LLVM 22)",
    "methodology": "5 interleaved rounds, fixed order, 2s sleeps between binaries, "
                   "nothing else launched by the harness; each binary does 3-warmup + "
                   "internal min-of-15 (mandelbrot 20); final value = min across rounds. "
                   "Runner: spmd-bench/measure.sh; log: spmd-bench/measure-log.txt.",
    "geomean_spmd_over_ispc_best": round(geomean, 4),
    "rows": rows,
    "raw_rounds": {b: dict(ms) for b, ms in runs.items()},
}
(ROOT / "spmd-bench" / "RESULTS.final.json").write_text(json.dumps(out, indent=2) + "\n")

for r in rows:
    print(f"{r['bench']+'/'+r['kernel']:28s} serial {r['serial_ms']:9.3f}  x4 {r['ispc_x4_ms']:9.3f}  "
          f"x8 {r['ispc_x8_ms']:9.3f}  spmd {r['spmd_ms']:9.3f}  ratio {r['spmd_over_ispc_best']:.3f}")
print(f"\ngeomean spmd/ISPC-best = {geomean:.4f}")
