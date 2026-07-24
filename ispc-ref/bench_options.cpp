// Baseline driver for the 'options' benchmark (Black-Scholes + binomial put).
// Pattern follows bench_mandel.cpp: fixed workload, 3 warmup + min-of-15 timed
// reps at whole-benchmark granularity, one CHECKSUM line, one MS line per timed
// variant, and reference-output dump when argv[1] gives a path prefix.
//
// Serial implementations adapted from ispc-bench/options_serial.cpp (Intel ISPC
// examples, BSD-3-Clause). Workload matches ISPC's examples/cpu/options default:
// nOptions = 128*1024, BINOMIAL_NUM = 64, S/X/T/r/v = 100/98/2/0.02/5.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define BINOMIAL_NUM 64

// ISPC non-task entry points (from options_ispc.h). Declared directly to avoid
// pulling in the task-variant declarations we do not link.
extern "C" void black_scholes_ispc(float Sa[], float Xa[], float Ta[], float ra[],
                                    float va[], float result[], int count);
extern "C" void binomial_put_ispc(float Sa[], float Xa[], float Ta[], float ra[],
                                   float va[], float result[], int count);

// ---- serial reference (adapted from options_serial.cpp) --------------------

static inline float CND(float X) {
    float L = fabsf(X);
    float k = 1.f / (1.f + 0.2316419f * L);
    float k2 = k * k;
    float k3 = k2 * k;
    float k4 = k2 * k2;
    float k5 = k3 * k2;
    const float invSqrt2Pi = 0.39894228040f;
    float w = (0.31938153f * k - 0.356563782f * k2 + 1.781477937f * k3 +
               -1.821255978f * k4 + 1.330274429f * k5);
    w *= invSqrt2Pi * expf(-L * L * .5f);
    if (X > 0.f)
        w = 1.f - w;
    return w;
}

static void black_scholes_serial(float *Sa, float *Xa, float *Ta, float *ra,
                                 float *va, float *result, int count) {
    for (int i = 0; i < count; ++i) {
        float S = Sa[i], X = Xa[i], T = Ta[i], r = ra[i], v = va[i];
        float d1 = (logf(S / X) + (r + v * v * .5f) * T) / (v * sqrtf(T));
        float d2 = d1 - v * sqrtf(T);
        result[i] = S * CND(d1) - X * expf(-r * T) * CND(d2);
    }
}

static void binomial_put_serial(float *Sa, float *Xa, float *Ta, float *ra,
                                float *va, float *result, int count) {
    float V[BINOMIAL_NUM];
    for (int i = 0; i < count; ++i) {
        float S = Sa[i], X = Xa[i], T = Ta[i], r = ra[i], v = va[i];
        float dt = T / BINOMIAL_NUM;
        float u = expf(v * sqrtf(dt));
        float d = 1.f / u;
        float disc = expf(r * dt);
        float Pu = (disc - d) / (u - d);
        for (int j = 0; j < BINOMIAL_NUM; ++j) {
            float upow = powf(u, (float)(2 * j - BINOMIAL_NUM));
            V[j] = std::max(0.f, X - S * upow);
        }
        for (int j = BINOMIAL_NUM - 1; j >= 0; --j)
            for (int k = 0; k < j; ++k)
                V[k] = ((1 - Pu) * V[k] + Pu * V[k + 1]) / disc;
        result[i] = V[0];
    }
}

// ---- timing harness --------------------------------------------------------

typedef void (*opt_fn)(float *, float *, float *, float *, float *, float *, int);

static double bench(opt_fn fn, float *S, float *X, float *T, float *r, float *v,
                    float *result, int count) {
    const int WARMUP = 3, REPS = 15;
    for (int i = 0; i < WARMUP; ++i)
        fn(S, X, T, r, v, result, count);
    double best = 1e30;
    for (int rep = 0; rep < REPS; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        fn(S, X, T, r, v, result, count);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

static double sum_arr(const float *a, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i)
        s += a[i];
    return s;
}

static void dump(const char *prefix, const char *suffix, const float *a, int n) {
    char path[1024];
    snprintf(path, sizeof(path), "%s%s", prefix, suffix);
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return; }
    fwrite(a, sizeof(float), n, f);
    fclose(f);
    fprintf(stderr, "wrote %s (%d floats)\n", path, n);
}

int main(int argc, char **argv) {
    const int nOptions = 128 * 1024; // 131072, ISPC example default
    float *S = new float[nOptions];
    float *X = new float[nOptions];
    float *T = new float[nOptions];
    float *r = new float[nOptions];
    float *v = new float[nOptions];
    float *res_bs = new float[nOptions];
    float *res_bin = new float[nOptions];
    for (int i = 0; i < nOptions; ++i) {
        S[i] = 100.f; X[i] = 98.f; T[i] = 2.f; r[i] = .02f; v[i] = 5.f;
    }

    // ISPC variants (linked object selects the target).
    double ms_bs_ispc  = bench(black_scholes_ispc, S, X, T, r, v, res_bs,  nOptions);
    double ms_bin_ispc = bench(binomial_put_ispc,  S, X, T, r, v, res_bin, nOptions);
    double sum_bs  = sum_arr(res_bs,  nOptions);
    double sum_bin = sum_arr(res_bin, nOptions);

    // Serial reference (into separate buffers so we can cross-check + dump).
    float *ser_bs  = new float[nOptions];
    float *ser_bin = new float[nOptions];
    double ms_bs_ser  = bench(black_scholes_serial, S, X, T, r, v, ser_bs,  nOptions);
    double ms_bin_ser = bench(binomial_put_serial,  S, X, T, r, v, ser_bin, nOptions);
    double sum_bs_ser  = sum_arr(ser_bs,  nOptions);
    double sum_bin_ser = sum_arr(ser_bin, nOptions);

    // Checksum = sum of both result arrays (ISPC), printed to 6 decimals.
    printf("CHECKSUM %.6f\n", sum_bs + sum_bin);
    printf("CHECKSUM_SERIAL %.6f\n", sum_bs_ser + sum_bin_ser);
    printf("MS black_scholes_ispc %.4f\n", ms_bs_ispc);
    printf("MS binomial_put_ispc %.4f\n", ms_bin_ispc);
    printf("MS black_scholes_serial %.4f\n", ms_bs_ser);
    printf("MS binomial_put_serial %.4f\n", ms_bin_ser);
    printf("SUM bs=%.6f binomial=%.6f (serial bs=%.6f binomial=%.6f)\n",
           sum_bs, sum_bin, sum_bs_ser, sum_bin_ser);

    if (argc > 1) {
        dump(argv[1], "_bs.bin",       res_bs,  nOptions);
        dump(argv[1], "_binomial.bin", res_bin, nOptions);
    }

    delete[] S; delete[] X; delete[] T; delete[] r; delete[] v;
    delete[] res_bs; delete[] res_bin; delete[] ser_bs; delete[] ser_bin;
    return 0;
}
