// Third-party derivation: the inlined serial reference is adapted from Intel's
// ISPC mandelbrot example, Copyright (c) Intel Corporation,
// SPDX-License-Identifier: BSD-3-Clause. See THIRD-PARTY.md.
//
// Benchmark driver for the ISPC "mandelbrot" example. Follows the shared
// pattern (steady_clock, 3 warmup + min-of-15 timed reps at whole-benchmark
// granularity, one CHECKSUM line and one MS line per variant, optional argv
// reference-output dump).
//
// Workload = ISPC example default: 768x512 image, maxIterations=256, view
// rectangle (-2,-1)..(1,1). Two variants timed: the linked ISPC NEON target
// (mandelbrot_ispc) and an inlined serial reference (adapted from
// ispc-bench/mandelbrot_serial.cpp). Both produce an int32 iteration-count
// image; the checksum is the integer sum over all pixels.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" void mandelbrot_ispc(float x0, float y0, float x1, float y1,
                                int32_t width, int32_t height,
                                int32_t maxIterations, int32_t *output);

// ---- Serial reference (from ispc-bench/mandelbrot_serial.cpp) ----
static int mandel(float c_re, float c_im, int count) {
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;
        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    return i;
}

static void mandelbrot_serial(float x0, float y0, float x1, float y1, int width,
                              int height, int maxIterations, int output[]) {
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; ++i) {
            float x = x0 + i * dx;
            float y = y0 + j * dy;
            output[j * width + i] = mandel(x, y, maxIterations);
        }
    }
}

static long long checksum(const int *buf, int n) {
    long long s = 0;
    for (int i = 0; i < n; i++) s += buf[i];
    return s;
}

int main(int argc, char **argv) {
    const int width = 768, height = 512, maxIter = 256;
    const int N = width * height;
    const int WARM = 3, REPS = 15;
    const float x0 = -2.f, y0 = -1.f, x1 = 1.f, y1 = 1.f;
    int *buf = new int[N];
    using clk = std::chrono::steady_clock;

    // ISPC (this binary links one target: neon-i32x4 or neon-i32x8)
    double bestISPC = 1e30;
    for (int r = 0; r < WARM + REPS; r++) {
        auto t0 = clk::now();
        mandelbrot_ispc(x0, y0, x1, y1, width, height, maxIter, buf);
        auto t1 = clk::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (r >= WARM && ms < bestISPC) bestISPC = ms;
    }
    long long csISPC = checksum(buf, N);

    // Serial reference
    int *sbuf = new int[N];
    double bestSerial = 1e30;
    for (int r = 0; r < WARM + REPS; r++) {
        auto t0 = clk::now();
        mandelbrot_serial(x0, y0, x1, y1, width, height, maxIter, sbuf);
        auto t1 = clk::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (r >= WARM && ms < bestSerial) bestSerial = ms;
    }
    long long csSerial = checksum(sbuf, N);

    printf("CHECKSUM %lld\n", csISPC);
    printf("MS mandelbrot_ispc %.3f\n", bestISPC);
    printf("MS mandelbrot_serial %.3f\n", bestSerial);
    fprintf(stderr, "# serial_checksum %lld  diff %lld\n", csSerial, csISPC - csSerial);

    // Dump ISPC reference output (raw int32, width*height) when a path is given.
    if (argc > 1) {
        FILE *f = fopen(argv[1], "wb");
        if (!f) { perror(argv[1]); return 1; }
        fwrite(buf, sizeof(int32_t), (size_t)N, f);
        fclose(f);
        fprintf(stderr, "# wrote reference %s (%d int32)\n", argv[1], N);
    }
    delete[] buf;
    delete[] sbuf;
    return 0;
}
