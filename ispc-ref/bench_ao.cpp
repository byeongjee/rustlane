// Baseline driver for the 'ao' (aobench) benchmark.
// Pattern follows bench_mandel.cpp: steady_clock, 3 warmup + min-of-15 timed
// reps at whole-benchmark granularity, one CHECKSUM line, one MS line per
// timed variant, optional reference-output dump when argv[1] is a path.
//
// Serial reference (ao_serial) comes from ispc-bench/ao_serial.cpp, linked in.
// NOTE: ISPC uses its stdlib RNG (seed_rng/frandom); the serial path uses
// drand48(). The two RNG streams differ, so ISPC and serial outputs are NOT
// bit-identical; their checksums are close but not equal. Each is reported
// separately. The dumped ref-out/ao_ispc.bin is the ISPC output.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "ao_ispc.h"

extern void ao_serial(int w, int h, int nsubsamples, float image[]);

// --- Thin serial-tasks shim -------------------------------------------------
// The kernel object also contains ao_ispc_tasks (uses launch[]), a global
// symbol that references ISPCLaunch/ISPCSync/ISPCAlloc. We only ever call the
// non-task ao_ispc, but the linker still needs those symbols resolved. This
// shim runs any launched tasks serially; it is never invoked in this driver
// (we do NOT call ao_ispc_tasks). We deliberately avoid ISPC's tasksys.
typedef void (*TaskFuncType)(void *data, int threadIndex, int threadCount,
                             int taskIndex, int taskCount, int taskIndex0,
                             int taskIndex1, int taskIndex2, int taskCount0,
                             int taskCount1, int taskCount2);
extern "C" {
void ISPCLaunch(void **, void *f, void *data, int c0, int c1, int c2) {
    TaskFuncType func = (TaskFuncType)f;
    int count = c0 * c1 * c2;
    for (int i = 0; i < count; i++)
        func(data, 0, 1, i, count, i % c0, (i / c0) % c1, i / (c0 * c1), c0, c1, c2);
}
void *ISPCAlloc(void **handlePtr, int64_t size, int32_t alignment) {
    void *p = nullptr;
    size_t a = alignment < (int)sizeof(void *) ? sizeof(void *) : (size_t)alignment;
    if (posix_memalign(&p, a, (size_t)size) != 0) p = nullptr;
    *handlePtr = p;
    return p;
}
void ISPCSync(void *handle) { free(handle); }
}

static double checksum(const float *img, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)img[i];
    return s;
}

int main(int argc, char **argv) {
    const int W = 512, H = 512, NS = 2;   // canonical aobench workload
    const int N = W * H * 3;
    const int WARM = 3, REPS = 15;
    float *fimg = new float[N];

    // --- ISPC (this binary links one target: neon-i32x4 or neon-i32x8) ---
    double bestISPC = 1e30;
    for (int r = 0; r < WARM + REPS; r++) {
        memset(fimg, 0, sizeof(float) * N);
        auto t0 = std::chrono::steady_clock::now();
        ispc::ao_ispc(W, H, NS, fimg);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (r >= WARM && ms < bestISPC) bestISPC = ms;
    }
    double csISPC = checksum(fimg, N);

    // --- Serial reference ---
    double bestSerial = 1e30;
    for (int r = 0; r < WARM + REPS; r++) {
        memset(fimg, 0, sizeof(float) * N);
        auto t0 = std::chrono::steady_clock::now();
        ao_serial(W, H, NS, fimg);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (r >= WARM && ms < bestSerial) bestSerial = ms;
    }
    double csSerial = checksum(fimg, N);

    printf("CHECKSUM ispc %.6f serial %.6f  (%dx%d ns=%d, N=%d floats)\n",
           csISPC, csSerial, W, H, NS, N);
    printf("MS ispc %.3f\n", bestISPC);
    printf("MS serial %.3f\n", bestSerial);

    // Dump ISPC reference output (raw float32 RGB, W*H*3) when a path is given.
    if (argc > 1) {
        memset(fimg, 0, sizeof(float) * N);
        ispc::ao_ispc(W, H, NS, fimg);
        FILE *fp = fopen(argv[1], "wb");
        if (!fp) { perror(argv[1]); return 1; }
        fwrite(fimg, sizeof(float), N, fp);
        fclose(fp);
        printf("Wrote reference output %s (%zu bytes)\n", argv[1], sizeof(float) * N);
    }
    delete[] fimg;
    return 0;
}
