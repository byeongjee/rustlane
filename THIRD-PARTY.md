# Third-Party Notices

This project is licensed as `MIT OR Apache-2.0` (Copyright (c) 2026 Byeongjee
Kang; see `LICENSE-MIT` and `LICENSE-APACHE`). The portions listed below are
bundled from, or derived from, third-party sources and **carry their own
upstream terms**, which continue to apply to that material. The per-file
copyright notices in the source files are authoritative; this document collects
them in one place and reproduces the license texts they require.

## Intel ISPC — BSD-3-Clause

- Upstream: <https://github.com/ispc/ispc>
- License: **BSD-3-Clause**
- Copyright: Intel Corporation

Most third-party material in this project derives from ISPC — both its example
programs (`examples/cpu/...`) and its standard library
(`stdlib/stdlib.ispc`, `stdlib/include/core.isph`). The `stdlib` ports cite
commit `e99a37840cd7d83c84e56e97a03eab6049b59fe7` (ISPC `main`) in their file
headers. The `examples`-derived C/C++ files were taken from an older ISPC
revision — most carry the pre-SPDX full BSD-3-Clause notice with copyright-year
ranges from `2010-2011` through `2010-2021` (the two mandelbrot C++ files
instead carry restored SPDX-style notices; see the table) — and that exact
source revision was not re-verified; the per-file notices are the authority for
those files.

The following is the ISPC license text, reproduced verbatim from
`LICENSE.txt` at the ISPC repository root. It governs every ISPC-derived item
inventoried in the three tables below and the bundled data files further down.

```
Copyright Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

### Verbatim / lightly-edited ISPC source files

These retain their original Intel copyright headers.

| File | Upstream | Notes |
| --- | --- | --- |
| `ispc-ref/ao.ispc` | `examples/cpu/aobench/ao.ispc` | Verbatim; Intel SPDX header; credits Syoyo Fujita's aobench (see aobench section) |
| `ispc-ref/mandelbrot.ispc` | `examples/cpu/mandelbrot/mandelbrot.ispc` | Verbatim; Intel SPDX header |
| `ispc-ref/options.ispc` | `examples/cpu/options/options.ispc` | Verbatim; Intel SPDX header |
| `ispc-ref/rt.ispc` | `examples/cpu/rt/rt.ispc` | Verbatim; Intel SPDX header |
| `ispc-ref/stencil.ispc` | `examples/cpu/stencil/stencil.ispc` | Verbatim; Intel SPDX header |
| `ispc-ref/volume.ispc` | `examples/cpu/volume_rendering/volume.ispc` | Verbatim; Intel SPDX header |
| `ispc-ref/options_defs.h` | `examples/cpu/options/options_defs.h` | Intel SPDX header |
| `ispc-ref/volume_serial.cpp` | `examples/cpu/volume_rendering/volume_serial.cpp` | Copy of `ispc-bench/volume_serial.cpp`; Intel BSD header |
| `ispc-bench/ao_serial.cpp` | `examples/cpu/aobench/ao_serial.cpp` | Intel BSD header (full text) |
| `ispc-bench/mandelbrot_serial.cpp` | `examples/cpu/mandelbrot/mandelbrot_serial.cpp` | Matches upstream byte-for-byte apart from a three-line provenance note; the Intel SPDX header (2010-2023) had been lost from this copy and was restored |
| `ispc-bench/options_serial.cpp` | `examples/cpu/options/options_serial.cpp` | Intel BSD header (full text) |
| `ispc-bench/rt_serial.cpp` | `examples/cpu/rt/rt_serial.cpp` | Intel BSD header (full text) |
| `ispc-bench/stencil_serial.cpp` | `examples/cpu/stencil/stencil_serial.cpp` | Intel BSD header (full text) |
| `ispc-bench/volume_serial.cpp` | `examples/cpu/volume_rendering/volume_serial.cpp` | Intel BSD header (full text) |
| `ispc-bench/timing.h` | ISPC `examples` timing helper | Intel BSD header (full text) |
| `ispc-bench/test-ao.cpp` | ISPC `aobench` example driver | Intel BSD header (full text) |
| `ispc-bench/test-options.cpp` | ISPC `options` example driver | Intel BSD header (full text) |
| `ispc-bench/test-rt.cpp` | ISPC `rt` example driver | Intel BSD header (full text) |
| `ispc-bench/test-stencil.cpp` | ISPC `stencil` example driver | Intel BSD header (full text) |
| `ispc-bench/test-volume.cpp` | ISPC `volume` example driver | Intel BSD header (full text) |
| `ispc-bench/test-mandelbrot.cpp` | adapted from `examples/cpu/mandelbrot/mandelbrot.cpp` (`writePPM` + harness) | Older writePPM variant; Intel header re-added with a yearless copyright because the exact source year could not be verified for this variant (its sibling `test-*.cpp` retain their original dated notices) |

### Adapted benchmark drivers (this project's harness, inlining ISPC-derived code)

These are original harness code that inline or link ISPC-derived serial
reference implementations and scene/data loaders. Each carries a header note
pointing here.

| File | Derived portion |
| --- | --- |
| `ispc-ref/bench_ao.cpp` | aobench scene + serial reference (linked from `ao_serial`) |
| `ispc-ref/bench_mandel.cpp` | inlined serial mandelbrot reference |
| `ispc-ref/bench_options.cpp` | serial Black-Scholes + binomial reference |
| `ispc-ref/bench_rt.cpp` | serial ray tracer + BVH/scene loaders (`examples/cpu/rt`) |
| `ispc-ref/bench_stencil.cpp` | inlined serial stencil reference |
| `ispc-ref/bench_volume.cpp` | serial volume reference + `loadCamera`/`loadVolume` loaders |

`ispc-ref/tasksys_shim.cpp` implements the ISPC task ABI
(`ISPCAlloc`/`ISPCLaunch`/`ISPCSync` and the `TaskFuncPtr` signature, which are
defined by ISPC and thus BSD-3-Clause); the serial implementation itself is
original to this project. Similarly, the `foreach_tiled!` lowering in
`rustlane-macros/src/rewrite.rs` reproduces the tile-span sizing and traversal
order of ISPC's `lGetSpans` (`src/stmt.cpp`, BSD-3-Clause) — a behavioral port
of that rule, noted here for completeness; the Rust implementation is
original.

### Rust ports of ISPC kernels and standard library

Algorithm and coefficient ports (re-implementations in Rust, not verbatim
source). Attribution here is for derivation of algorithms and numeric
coefficients. Each file carries a header note pointing here.

| File | Ported from |
| --- | --- |
| `rustlane-core/src/math.rs` | ISPC `stdlib/stdlib.ispc` default-precision polynomials for `exp`, `log`, `sin`, `cos`, `pow`. The thin hardware wrappers (`sqrt`/`abs`/`min`/`max`/`floor`/`ceil`/`round`/`clamp`/`lerp`) and the NEON reciprocal/rsqrt estimates are original. See the syrah note below. |
| `rustlane-core/src/rng.rs` | ISPC `stdlib/stdlib.ispc` `random`/`frandom`/`seed_rng` and `struct RNGState` from `stdlib/include/core.isph` (LFSR113 / combined-Tausworthe) |
| `rustlane-bench/src/bin/ao.rs` | `ispc-ref/ao.ispc` (aobench) |
| `rustlane-bench/src/bin/mandelbrot.rs` | `ispc-ref/mandelbrot.ispc` |
| `rustlane-bench/src/bin/mandelbrot_export.rs` | Same `mandel` kernel as `mandelbrot.rs` (from `ispc-ref/mandelbrot.ispc`), driven through the `#[export]` entry point |
| `rustlane-bench/src/bin/options.rs` | `ispc-ref/options.ispc` (incl. the CND polynomial) |
| `rustlane-bench/src/bin/rt.rs` | `ispc-ref/rt.ispc` |
| `rustlane-bench/src/bin/stencil.rs` | `ispc-ref/stencil.ispc` |
| `rustlane-bench/src/bin/volume.rs` | `ispc-ref/volume.ispc` |

### Generated files

`ispc-ref/mandelbrot_ispc.h` is emitted by the ISPC compiler from
`mandelbrot.ispc` (it carries the "automatically generated by the ispc
compiler" banner) and is a build artifact of BSD-3-Clause source. Other
ISPC-generated headers (`*_ispc.h`, `*_ispc_x8.h`), object files (`*.o`),
compiled benchmark binaries, and the reference outputs under
`ispc-ref/ref-out/*.bin` are build artifacts excluded by `.gitignore`
(`ref-out/*.bin` are this project's own generated dumps, not third-party).

## aobench — Syoyo Fujita (BSD-3-Clause)

- Author: Syoyo Fujita
- Upstream: <https://github.com/syoyo/aobench> (originally hosted at the
  now-defunct `http://code.google.com/p/aobench`)
- License: **BSD-3-Clause**

The `ao` benchmark reached this project through ISPC: ISPC's
`examples/cpu/aobench/ao.ispc` is itself "Based on Syoyo Fujita's aobench", and
`ispc-ref/ao.ispc` plus its Rust port `rustlane-bench/src/bin/ao.rs` derive from
that. ISPC's copy carries Intel's BSD-3-Clause header (reproduced above) and
retains the Syoyo Fujita credit line; both are preserved in this project. There
is no license conflict — BSD-3-Clause over BSD-3-Clause.

The upstream aobench `COPYING` file is reproduced below for completeness. Its
copyright line reads `Copyright 2024 Syoyo Fujita` — that is the notice as the
current upstream `COPYING` states it (the project itself dates to roughly 2009);
the notice is reproduced as written.

```
Copyright 2024 Syoyo Fujita

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
“AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## syrah — Solomon Boulos

- Author: Solomon Boulos
- Upstream: <https://github.com/boulos/syrah>

The transcendental polynomial coefficients that ISPC's standard library uses
(and therefore the ports in `rustlane-core/src/math.rs`) originate from syrah, as
credited in ISPC's `stdlib.ispc` (`// Solomon Boulos's "syrah"`). These
coefficients entered this project through ISPC's BSD-3-Clause standard library,
not directly from syrah; the credit is retained as a courtesy and for accuracy.

## Bundled data files

These binary inputs live under `ispc-bench/`. They cannot carry an in-file
header, so they are documented here. All four are shipped by Intel in the ISPC
`examples/` tree, which Intel distributes under the repo-wide BSD-3-Clause
license (reproduced above). ISPC provides **no per-file provenance, README, or
NOTICE** for these data files.

| File | Shipped by ISPC in | Nature |
| --- | --- | --- |
| `ispc-bench/sponza.bvh` | `examples/cpu/rt/` | Preprocessed BVH of a "Sponza" scene (~37k triangles) — see note below |
| `ispc-bench/sponza.camera` | `examples/cpu/rt/` | 34 floats: camera / projection matrices (functional numeric config, no expressive content) |
| `ispc-bench/density_lowres.vol` | `examples/cpu/volume_rendering/` | ASCII-header (`48 64 48`) + float density grid — a procedural/simulated smoke field |
| `ispc-bench/camera.dat` | `examples/cpu/volume_rendering/` | 4×4 matrices (functional numeric config, no expressive content) |

### `sponza.bvh` / `sponza.camera` — provenance uncertainty

This is the one item with a provenance gap, stated here rather than
overclaimed:

- The `.bvh` is a preprocessed bounding-volume hierarchy, not a source model.
  Parsing it gives ~121,149 BVH nodes and ~37,380 triangles. That triangle
  count **rules out the Crytek Sponza** (~262k triangles, 2010) and places the
  geometry in the **original pre-Crytek Sponza lineage** — Marko Dabrovic's
  "Atrium Sponza Palace" (2002), which was informally donated for public use in
  radiosity research with credit requested.
- The exact source model file, and any decimation ISPC applied, are
  **undocumented**. "Original Dabrovic lineage" is a medium-high-confidence
  inference from the triangle count; the precise upstream file is unknown.
- Consequently the terms actually applicable to this model data, as ISPC
  distributes it, are **undocumented / unclear**. Intel asserts BSD-3-Clause
  over its whole repository, but the geometry is a preprocessed derivative of a
  third-party model whose original grant was informal, not a formal OSI/CC
  license.
- **Do not** label this file "Crytek Sponza" or apply CC BY 3.0 — the file is
  not Crytek's model, and doing so would be an overclaim.

Chosen disposition: this project **keeps the files and attributes them** as
above. `sponza.camera` is a pure numeric matrix with no copyright concern. The
practical risk is low — the original model was freely donated, and the BVH is
geometry only (no textures or materials) — but because ISPC documents no terms,
this project attributes explicitly to Marko Dabrovic (original model) and
Intel/ISPC (the BVH as redistributed). A cleaner alternative for downstream
redistributors who wish to avoid shipping undocumented third-party geometry is
to drop `sponza.bvh` from their tree and fetch it on demand from the ISPC
`examples/cpu/rt/` directory, since it is only a benchmark input.

### `density_lowres.vol` / `camera.dat`

Lowest risk. `density_lowres.vol` is a small procedural density grid with no
evidence of any third-party origin, and `camera.dat` is a plain 4×4 matrix
(non-expressive numeric config). Absent any external attribution trail, these
are best treated as ISPC-authored example data under BSD-3-Clause (Intel); this
is an inference from the absence of any third-party trail, not a documented
fact, but the legal risk is low and no CC/model license is implicated.

## Note for binary redistributors (BSD-3-Clause clause 2)

BSD-3-Clause clause 2 requires that redistributions **in binary form** reproduce
the applicable copyright notice, the list of conditions, and the disclaimer "in
the documentation and/or other materials provided with the distribution."

If you distribute compiled binaries built from any of the ISPC-derived material
(or the aobench-derived `ao` code) inventoried above, you satisfy clause 2 by
including this `THIRD-PARTY.md` file — which reproduces Intel's and Syoyo
Fujita's copyright notices, conditions, and disclaimers — in your product's
documentation or accompanying materials. Retaining this file alongside the
binaries is sufficient; no additional per-binary notice is required.
