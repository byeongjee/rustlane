#include <algorithm>
#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include "timing.h"

extern void mandelbrot_serial(float x0, float y0, float x1, float y1, int width, int height, int maxIterations,
                              int output[]);

/* Write a PPM image file with the image of the Mandelbrot set */
static void writePPM(int *buf, int width, int height, const char *fn) {
    FILE *debug_fp = fopen("debug.txt", "w");
    FILE *fp = fopen(fn, "wb");
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (int i = 0; i < width * height; ++i) {
        // Map the iteration count to colors by just alternating between
        // two greys.
        char c = (buf[i] & 0x1) ? (char)240 : 20;
        for (int j = 0; j < 3; ++j) {
          fprintf(debug_fp, "%d ", (int)c);
            fputc(c, fp);
        }
        fputc('\n', debug_fp);
    }
    fclose(fp);
    fclose(debug_fp);
    printf("Wrote image file %s\n", fn);
}

int main(int argc, char *argv[]) {
    static unsigned int test_iterations[] = {3, 3};
    unsigned int width = 768;
    unsigned int height = 512;
    float x0 = -2;
    float x1 = 1;
    float y0 = -1;
    float y1 = 1;

    if (argc > 1) {
        if (strncmp(argv[1], "--scale=", 8) == 0) {
            float scale = atof(argv[1] + 8);
            width *= scale;
            height *= scale;
        }
    }
    if ((argc == 3) || (argc == 4)) {
        for (int i = 0; i < 2; i++) {
            test_iterations[i] = atoi(argv[argc - 2 + i]);
        }
    }

    int maxIterations = 256;
    int *buf = new int[width * height];

#if 0
    //
    // Compute the image using the ispc implementation; report the minimum
    // time of three runs.
    //
    double minISPC = 1e30;
    for (unsigned int i = 0; i < test_iterations[0]; ++i) {
        reset_and_start_timer();
        mandelbrot_ispc(x0, y0, x1, y1, width, height, maxIterations, buf);
        double dt = get_elapsed_mcycles();
        printf("@time of ISPC run:\t\t\t[%.3f] million cycles\n", dt);
        minISPC = std::min(minISPC, dt);
    }

    printf("[mandelbrot ispc]:\t\t[%.3f] million cycles\n", minISPC);
    writePPM(buf, width, height, "mandelbrot-ispc.ppm");
#endif

    // Clear out the buffer
    for (unsigned int i = 0; i < width * height; ++i)
        buf[i] = 0;

    //
    // And run the serial implementation 3 times, again reporting the
    // minimum time.
    //
    double minSerial = 1e30;
    for (unsigned int i = 0; i < test_iterations[1]; ++i) {
        reset_and_start_timer();
        mandelbrot_serial(x0, y0, x1, y1, width, height, maxIterations, buf);
        double dt = get_elapsed_mcycles();
        printf("@time of serial run:\t\t\t[%.3f] million cycles\n", dt);
        minSerial = std::min(minSerial, dt);
    }

    printf("[mandelbrot serial]:\t\t[%.3f] million cycles\n", minSerial);
    writePPM(buf, width, height, "mandelbrot-serial.ppm");

    //printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minSerial / minISPC);

    return 0;
}
