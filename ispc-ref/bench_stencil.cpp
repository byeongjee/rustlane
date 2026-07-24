// Benchmark driver for the ISPC 'stencil' example (3-D 7-point-ish iterated
// leapfrog stencil). Follows the bench_mandel.cpp pattern: fixed workload,
// 3 warmup + min-of-15 timed reps at whole-benchmark granularity, one CHECKSUM
// line, one MS line per timed variant, optional argv reference-output dump.
//
// Workload = ISPC example default (examples/cpu/stencil/stencil.cpp):
//   Nx=Ny=Nz=256, width=4, iteration range t=[0,6),
//   coef={0.5,-0.25,0.125,-0.0625}, InitData() as in the example.
//
// The kernel writes alternate between the two buffers (Aeven/Aodd) per
// iteration, so buffers mutate in place. We re-initialize before every timed
// rep (outside the timed region) so each rep does identical work and the
// final-buffer checksum is deterministic.
//
// Serial reference below is adapted verbatim from ispc-bench/stencil_serial.cpp
// (Intel Corporation, BSD-3-Clause).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Non-task ISPC entry point (avoids pulling in ISPC's tasksys).
extern "C" void loop_stencil_ispc(int t0, int t1, int x0, int x1, int y0, int y1,
                                  int z0, int z1, int Nx, int Ny, int Nz,
                                  const float coef[], const float vsq[],
                                  float Aeven[], float Aodd[]);

// The kernel object also contains loop_stencil_ispc_tasks (never called here),
// whose task ABI references these symbols. We provide never-reached stubs so we
// avoid linking ISPC's real tasksys; abort() guards against accidental use.
extern "C" {
void *ISPCAlloc(void **, int64_t, int32_t) { abort(); }
void ISPCLaunch(void **, void *, void *, int, int, int) { abort(); }
void ISPCSync(void *) { abort(); }
}

// ---- Serial reference (from ispc-bench/stencil_serial.cpp) ----
static void stencil_step_serial(int x0, int x1, int y0, int y1, int z0, int z1,
                                int Nx, int Ny, int Nz,
                                const float *__restrict__ coef,
                                const float *__restrict__ vsq,
                                const float *__restrict__ Ain,
                                float *__restrict__ Aout) {
    int Nxy = Nx * Ny;
    for (int z = z0; z < z1; ++z) {
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                int index = (z * Nxy) + (y * Nx) + x;
#define A_cur(x, y, z) Ain[index + (x) + ((y)*Nx) + ((z)*Nxy)]
#define A_next(x, y, z) Aout[index + (x) + ((y)*Nx) + ((z)*Nxy)]
                float div =
                    coef[0] * A_cur(0, 0, 0) +
                    coef[1] * (A_cur(+1, 0, 0) + A_cur(-1, 0, 0) + A_cur(0, +1, 0) +
                               A_cur(0, -1, 0) + A_cur(0, 0, +1) + A_cur(0, 0, -1)) +
                    coef[2] * (A_cur(+2, 0, 0) + A_cur(-2, 0, 0) + A_cur(0, +2, 0) +
                               A_cur(0, -2, 0) + A_cur(0, 0, +2) + A_cur(0, 0, -2)) +
                    coef[3] * (A_cur(+3, 0, 0) + A_cur(-3, 0, 0) + A_cur(0, +3, 0) +
                               A_cur(0, -3, 0) + A_cur(0, 0, +3) + A_cur(0, 0, -3));
                A_next(0, 0, 0) = 2 * A_cur(0, 0, 0) - A_next(0, 0, 0) + vsq[index] * div;
#undef A_cur
#undef A_next
            }
        }
    }
}

static void loop_stencil_serial(int t0, int t1, int x0, int x1, int y0, int y1,
                                int z0, int z1, int Nx, int Ny, int Nz,
                                const float *__restrict__ coef,
                                const float *__restrict__ vsq,
                                float *__restrict__ Aeven,
                                float *__restrict__ Aodd) {
    for (int t = t0; t < t1; ++t) {
        if ((t & 1) == 0)
            stencil_step_serial(x0, x1, y0, y1, z0, z1, Nx, Ny, Nz, coef, vsq, Aeven, Aodd);
        else
            stencil_step_serial(x0, x1, y0, y1, z0, z1, Nx, Ny, Nz, coef, vsq, Aodd, Aeven);
    }
}

// ---- InitData (from the ISPC example) ----
static void InitData(int Nx, int Ny, int Nz, float *A0, float *A1, float *vsq) {
    int offset = 0;
    for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
            for (int x = 0; x < Nx; ++x, ++offset) {
                A0[offset] = (x < Nx / 2) ? x / float(Nx) : y / float(Ny);
                A1[offset] = 0;
                vsq[offset] = x * y * z / float(Nx * Ny * Nz);
            }
}

int main(int argc, char **argv) {
    const int Nx = 256, Ny = 256, Nz = 256, width = 4;
    const int t0 = 0, t1 = 6;
    const long long N = (long long)Nx * Ny * Nz;
    const float coef[4] = {0.5f, -0.25f, 0.125f, -0.0625f};
    const int warmup = 3, reps = 15;

    float *Ai0 = new float[N], *Ai1 = new float[N];
    float *As0 = new float[N], *As1 = new float[N];
    float *vsq = new float[N];

    using clk = std::chrono::steady_clock;

    // ISPC (linked NEON target)
    double best_ispc = 1e30;
    for (int r = 0; r < warmup + reps; r++) {
        InitData(Nx, Ny, Nz, Ai0, Ai1, vsq);
        auto c0 = clk::now();
        loop_stencil_ispc(t0, t1, width, Nx - width, width, Ny - width, width, Nz - width,
                          Nx, Ny, Nz, coef, vsq, Ai0, Ai1);
        auto c1 = clk::now();
        double ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
        if (r >= warmup && ms < best_ispc) best_ispc = ms;
    }

    // Serial
    double best_serial = 1e30;
    for (int r = 0; r < warmup + reps; r++) {
        InitData(Nx, Ny, Nz, As0, As1, vsq);
        auto c0 = clk::now();
        loop_stencil_serial(t0, t1, width, Nx - width, width, Ny - width, width, Nz - width,
                            Nx, Ny, Nz, coef, vsq, As0, As1);
        auto c1 = clk::now();
        double ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
        if (r >= warmup && ms < best_serial) best_serial = ms;
    }

    // Buffer written on the final iteration (t=t1-1): even t writes Aodd, odd writes Aeven.
    const int last_t = t1 - 1;
    float *ispc_final = ((last_t & 1) == 0) ? Ai1 : Ai0;
    float *serial_final = ((last_t & 1) == 0) ? As1 : As0;

    double sum_ispc = 0.0, sum_serial = 0.0;
    for (long long i = 0; i < N; i++) {
        sum_ispc += ispc_final[i];
        sum_serial += serial_final[i];
    }

    printf("CHECKSUM %.6f\n", sum_ispc);
    printf("MS stencil_ispc %.3f\n", best_ispc);
    printf("MS stencil_serial %.3f\n", best_serial);
    double reld = (sum_serial != 0.0) ? (sum_ispc - sum_serial) / sum_serial : (sum_ispc - sum_serial);
    fprintf(stderr, "# serial_checksum %.6f  reldiff %.3e\n", sum_serial, reld);

    if (argc > 1) {
        FILE *f = fopen(argv[1], "wb");
        if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
        fwrite(ispc_final, sizeof(float), (size_t)N, f);
        fclose(f);
        fprintf(stderr, "# wrote reference %s (%lld floats)\n", argv[1], N);
    }
    return 0;
}
