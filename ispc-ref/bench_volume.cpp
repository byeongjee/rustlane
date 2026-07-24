// Third-party derivation: the serial reference and the loadCamera/loadVolume
// loaders are adapted from Intel's ISPC volume example, Copyright (c) Intel
// Corporation, SPDX-License-Identifier: BSD-3-Clause. See THIRD-PARTY.md.
//
// Benchmark driver for the ISPC "volume" example (ray-marched volume rendering).
// Pattern mirrors bench_mandel.cpp: fixed workload, 3 warmup + min-of-15 timed
// reps at whole-benchmark granularity, one CHECKSUM line and one MS line per
// variant (serial + linked ISPC target). Dumps the serial reference image to a
// binary file when argv[1] is given.
//
// Loaders (loadCamera/loadVolume) adapted from ispc-bench/test-volume.cpp.
// Serial implementation compiled from ispc-ref/volume_serial.cpp (copied from
// ispc-bench/volume_serial.cpp). ISPC kernel entry is the non-task volume_ispc;
// the never-called task-ABI stubs below satisfy the launch[] references that the
// unused volume_ispc_tasks entry leaves in the object (no ISPC tasksys linked).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Data paths relative to the ispc-ref working directory (where the harness
// runs the benchmark binaries from; see measure.sh and the Makefile refs rule).
static const char *kCameraPath = "../ispc-bench/camera.dat";
static const char *kVolumePath = "../ispc-bench/density_lowres.vol";

extern "C" void volume_ispc(float *density, int32_t *nVoxels,
                            const float raster2camera[][4],
                            const float camera2world[][4],
                            int32_t width, int32_t height, float *image);

extern void volume_serial(float *density, int *nVoxels,
                          const float *raster2camera_ptr,
                          const float *camera2world_ptr,
                          int width, int height, float *image);

// ISPC task-ABI stubs: referenced by the unused volume_ispc_tasks entry in the
// kernel object but never called here. Abort if ever reached.
extern "C" void ISPCLaunch(void **, void *, void *, int, int, int) { abort(); }
extern "C" void *ISPCAlloc(void **, int64_t, int32_t) { abort(); }
extern "C" void ISPCSync(void *) { abort(); }

static void loadCamera(const char *fn, int *width, int *height,
                       float raster2camera[4][4], float camera2world[4][4]) {
    FILE *f = fopen(fn, "r");
    if (!f) { perror(fn); exit(1); }
    if (fscanf(f, "%d %d", width, height) != 2) {
        fprintf(stderr, "Unexpected end of file in camera file\n"); exit(1);
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (fscanf(f, "%f", &raster2camera[i][j]) != 1) {
                fprintf(stderr, "Unexpected end of file in camera file\n"); exit(1);
            }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (fscanf(f, "%f", &camera2world[i][j]) != 1) {
                fprintf(stderr, "Unexpected end of file in camera file\n"); exit(1);
            }
    fclose(f);
}

static float *loadVolume(const char *fn, int n[3]) {
    FILE *f = fopen(fn, "r");
    if (!f) { perror(fn); exit(1); }
    if (fscanf(f, "%d %d %d", &n[0], &n[1], &n[2]) != 3) {
        fprintf(stderr, "Couldn't find resolution at start of density file\n"); exit(1);
    }
    int count = n[0] * n[1] * n[2];
    float *v = new float[count];
    for (int i = 0; i < count; ++i)
        if (fscanf(f, "%f", &v[i]) != 1) {
            fprintf(stderr, "Unexpected end of file at %d'th density value\n", i); exit(1);
        }
    fclose(f);
    return v;
}

static double checksum(const float *img, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)img[i];
    return s;
}

int main(int argc, char **argv) {
    const int warmup = 3, reps = 15;

    int width, height;
    float raster2camera[4][4], camera2world[4][4];
    loadCamera(kCameraPath, &width, &height, raster2camera, camera2world);

    int n[3];
    float *density = loadVolume(kVolumePath, n);

    const int npix = width * height;
    float *img_serial = new float[npix];
    float *img_ispc = new float[npix];

    // --- ISPC (linked target) ---
    for (int r = 0; r < warmup; ++r)
        volume_ispc(density, n, raster2camera, camera2world, width, height, img_ispc);
    double best_ispc = 1e30;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        volume_ispc(density, n, raster2camera, camera2world, width, height, img_ispc);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best_ispc) best_ispc = ms;
    }

    // --- Serial ---
    for (int r = 0; r < warmup; ++r)
        volume_serial(density, n, &raster2camera[0][0], &camera2world[0][0], width, height, img_serial);
    double best_serial = 1e30;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        volume_serial(density, n, &raster2camera[0][0], &camera2world[0][0], width, height, img_serial);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best_serial) best_serial = ms;
    }

    printf("CHECKSUM serial %.6f\n", checksum(img_serial, npix));
    printf("CHECKSUM ispc %.6f\n", checksum(img_ispc, npix));
    printf("MS serial %.3f\n", best_serial);
    printf("MS ispc %.3f\n", best_ispc);
    printf("workload %dx%d image, %dx%dx%d volume\n", width, height, n[0], n[1], n[2]);

    // Dump serial reference image (width*height floats, row-major) when a path
    // is provided.
    if (argc > 1) {
        FILE *fp = fopen(argv[1], "wb");
        if (!fp) { perror(argv[1]); return 1; }
        fwrite(img_serial, sizeof(float), npix, fp);
        fclose(fp);
        printf("Wrote reference output %s (%zu bytes)\n", argv[1], sizeof(float) * (size_t)npix);
    }

    delete[] density;
    delete[] img_serial;
    delete[] img_ispc;
    return 0;
}
