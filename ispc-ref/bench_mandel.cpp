#include <chrono>
#include <cstdio>
#include <cstdlib>
extern "C" void mandelbrot_ispc(float, float, float, float, int32_t, int32_t, int32_t, int32_t*);
int main(int argc, char** argv) {
    int width = 768, height = 512, maxIter = 256, reps = 20;
    int* buf = new int[width * height];
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        auto t0 = std::chrono::steady_clock::now();
        mandelbrot_ispc(-2.f, -1.f, 1.f, 1.f, width, height, maxIter, buf);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    long long sum = 0; for (int i = 0; i < width*height; i++) sum += buf[i];
    printf("%s: %.3f ms  (checksum %lld)\n", argv[0], best, sum);
    return 0;
}
