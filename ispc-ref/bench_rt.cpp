// bench_rt.cpp - ISPC + serial baseline driver for the 'rt' BVH ray tracer.
// Follows the bench_mandel.cpp pattern: fixed workload (sponza scene),
// 3 warmup + min-of-15 timed reps at whole-benchmark granularity,
// one CHECKSUM line + one MS line per timed variant, optional ref dump.
//
// Serial reference implementation and scene loaders are adapted from ISPC's
// official example: examples/cpu/rt (rt_serial.cpp + rt.cpp). Struct layouts
// are byte-identical to the ISPC-generated ones in rt_ispc.h, so the loaded
// arrays are shared between serial and ISPC via reinterpret_cast.
#include "rt_ispc.h"       // ispc:: LinearBVHNode, Triangle; extern "C" raytrace_ispc
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Serial reference implementation (adapted verbatim from ISPC rt_serial.cpp)
// ---------------------------------------------------------------------------
namespace srt {

struct float3 {
    float3() {}
    float3(float xx, float yy, float zz) { x = xx; y = yy; z = zz; }
    float3 operator*(float f) const { return float3(x * f, y * f, z * f); }
    float3 operator-(const float3 &f2) const { return float3(x - f2.x, y - f2.y, z - f2.z); }
    float3 operator*(const float3 &f2) const { return float3(x * f2.x, y * f2.y, z * f2.z); }
    float x, y, z;
    float pad; // match padding/alignment of ispc version
} __attribute__((aligned(16)));

struct Ray {
    float3 origin, dir, invDir;
    unsigned int dirIsNeg[3];
    float mint, maxt;
    int hitId;
};

struct Triangle {
    float p[3][4]; // extra float pad after each vertex
    int32_t id;
    int32_t pad[3]; // make 16 x 32-bits
};

struct LinearBVHNode {
    float bounds[2][3];
    int32_t offset; // primitives for leaf, second child for interior
    uint8_t nPrimitives;
    uint8_t splitAxis;
    uint16_t pad;
};

__attribute__((always_inline)) inline float3 Cross(const float3 &v1, const float3 &v2) {
    float v1x = v1.x, v1y = v1.y, v1z = v1.z;
    float v2x = v2.x, v2y = v2.y, v2z = v2.z;
    float3 ret;
    ret.x = (v1y * v2z) - (v1z * v2y);
    ret.y = (v1z * v2x) - (v1x * v2z);
    ret.z = (v1x * v2y) - (v1y * v2x);
    return ret;
}

__attribute__((always_inline)) inline float Dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__attribute__((always_inline)) static void generateRay(float *__restrict__ raster2camera_ptr,
                                                        float *__restrict__ camera2world_ptr, float x, float y,
                                                        Ray &ray) {
    auto &raster2camera = *(float(*)[4][4])(raster2camera_ptr);
    auto &camera2world = *(float(*)[4][4])(camera2world_ptr);

    ray.mint = 0.f;
    ray.maxt = 1e30f;
    ray.hitId = 0;

    float camx = raster2camera[0][0] * x + raster2camera[0][1] * y + raster2camera[0][3];
    float camy = raster2camera[1][0] * x + raster2camera[1][1] * y + raster2camera[1][3];
    float camz = raster2camera[2][3];
    float camw = raster2camera[3][3];
    camx /= camw;
    camy /= camw;
    camz /= camw;

    ray.dir.x = camera2world[0][0] * camx + camera2world[0][1] * camy + camera2world[0][2] * camz;
    ray.dir.y = camera2world[1][0] * camx + camera2world[1][1] * camy + camera2world[1][2] * camz;
    ray.dir.z = camera2world[2][0] * camx + camera2world[2][1] * camy + camera2world[2][2] * camz;

    ray.origin.x = camera2world[0][3] / camera2world[3][3];
    ray.origin.y = camera2world[1][3] / camera2world[3][3];
    ray.origin.z = camera2world[2][3] / camera2world[3][3];

    ray.invDir.x = 1.f / ray.dir.x;
    ray.invDir.y = 1.f / ray.dir.y;
    ray.invDir.z = 1.f / ray.dir.z;

    ray.dirIsNeg[0] = (ray.invDir.x < 0) ? 1 : 0;
    ray.dirIsNeg[1] = (ray.invDir.y < 0) ? 1 : 0;
    ray.dirIsNeg[2] = (ray.invDir.z < 0) ? 1 : 0;
}

__attribute__((always_inline)) static inline bool BBoxIntersect(const float bounds[2][3], const Ray &ray) {
    float3 bounds0(bounds[0][0], bounds[0][1], bounds[0][2]);
    float3 bounds1(bounds[1][0], bounds[1][1], bounds[1][2]);
    float t0 = ray.mint, t1 = ray.maxt;

    float3 tNear = (bounds0 - ray.origin) * ray.invDir;
    float3 tFar = (bounds1 - ray.origin) * ray.invDir;
    if (tNear.x > tFar.x) { float tmp = tNear.x; tNear.x = tFar.x; tFar.x = tmp; }
    t0 = std::max(tNear.x, t0);
    t1 = std::min(tFar.x, t1);
    if (tNear.y > tFar.y) { float tmp = tNear.y; tNear.y = tFar.y; tFar.y = tmp; }
    t0 = std::max(tNear.y, t0);
    t1 = std::min(tFar.y, t1);
    if (tNear.z > tFar.z) { float tmp = tNear.z; tNear.z = tFar.z; tFar.z = tmp; }
    t0 = std::max(tNear.z, t0);
    t1 = std::min(tFar.z, t1);
    return (t0 <= t1);
}

} // namespace srt

namespace srt {

__attribute__((always_inline)) inline bool TriIntersect(const Triangle &tri, Ray &ray) {
    float3 p0(tri.p[0][0], tri.p[0][1], tri.p[0][2]);
    float3 p1(tri.p[1][0], tri.p[1][1], tri.p[1][2]);
    float3 p2(tri.p[2][0], tri.p[2][1], tri.p[2][2]);
    float3 e1 = p1 - p0;
    float3 e2 = p2 - p0;

    float3 s1 = Cross(ray.dir, e2);
    float divisor = Dot(s1, e1);
    if (divisor == 0.)
        return false;
    float invDivisor = 1.f / divisor;

    float3 d = ray.origin - p0;
    float b1 = Dot(d, s1) * invDivisor;
    if (b1 < 0. || b1 > 1.)
        return false;

    float3 s2 = Cross(d, e1);
    float b2 = Dot(ray.dir, s2) * invDivisor;
    if (b2 < 0. || b1 + b2 > 1.)
        return false;

    float t = Dot(e2, s2) * invDivisor;
    if (t < ray.mint || t > ray.maxt)
        return false;

    ray.maxt = t;
    ray.hitId = tri.id;
    return true;
}

static __attribute__((always_inline)) bool BVHIntersect(const LinearBVHNode *__restrict__ nodes,
                                                         const Triangle *__restrict__ tris, Ray &r) {
    Ray ray = r;
    bool hit = false;
    int todoOffset = 0, nodeNum = 0;
    int todo[64];

    while (true) {
        const LinearBVHNode &node = nodes[nodeNum];
        if (BBoxIntersect(node.bounds, ray)) {
            unsigned int nPrimitives = node.nPrimitives;
            if (nPrimitives > 0) {
                unsigned int primitivesOffset = node.offset;
                for (unsigned int i = 0; i < nPrimitives; ++i) {
                    if (TriIntersect(tris[primitivesOffset + i], ray))
                        hit = true;
                }
                if (todoOffset == 0)
                    break;
                nodeNum = todo[--todoOffset];
            } else {
                if (r.dirIsNeg[node.splitAxis]) {
                    todo[todoOffset++] = nodeNum + 1;
                    nodeNum = node.offset;
                } else {
                    todo[todoOffset++] = node.offset;
                    nodeNum = nodeNum + 1;
                }
            }
        } else {
            if (todoOffset == 0)
                break;
            nodeNum = todo[--todoOffset];
        }
    }
    r.maxt = ray.maxt;
    r.hitId = ray.hitId;
    return hit;
}

static void raytrace_serial(int width, int height, int baseWidth, int baseHeight,
                            float *__restrict__ raster2camera_ptr, float *__restrict__ camera2world_ptr,
                            float *__restrict__ image, int *__restrict__ id,
                            const LinearBVHNode *__restrict__ nodes, const Triangle *__restrict__ triangles) {
    float widthScale = float(baseWidth) / float(width);
    float heightScale = float(baseHeight) / float(height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Ray ray;
            generateRay(raster2camera_ptr, camera2world_ptr, x * widthScale, y * heightScale, ray);
            BVHIntersect(nodes, triangles, ray);
            int offset = y * width + x;
            image[offset] = ray.maxt;
            id[offset] = ray.hitId;
        }
    }
}

} // namespace srt

// ---------------------------------------------------------------------------
// Scene loaders (adapted from ISPC rt.cpp main()) + benchmark harness
// ---------------------------------------------------------------------------
#ifndef SCENE_BASE
#define SCENE_BASE "/Users/byeongjee/side/rust-ispc/ispc-bench/sponza"
#endif

#define READ(var, n)                                                                                                   \
    if (fread(&(var), sizeof(var), (n), f) != (size_t)(n)) {                                                           \
        fprintf(stderr, "Unexpected EOF reading scene file\n");                                                        \
        return 1;                                                                                                      \
    } else /* eat ; */

static double checksum_image(const float *image, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i)
        s += (double)image[i];
    return s;
}

int main(int argc, char *argv[]) {
    const char *base = SCENE_BASE;
    const char *dumpPath = (argc > 1) ? argv[1] : nullptr; // optional ref-output dump
    char fnbuf[1200];

    // --- camera ---
    int baseWidth = 0, baseHeight = 0;
    float camera2world[4][4], raster2camera[4][4];
    snprintf(fnbuf, sizeof(fnbuf), "%s.camera", base);
    FILE *f = fopen(fnbuf, "rb");
    if (!f) { perror(fnbuf); return 1; }
    READ(baseWidth, 1);
    READ(baseHeight, 1);
    READ(camera2world[0][0], 16);
    READ(raster2camera[0][0], 16);
    fclose(f);

    // --- BVH: nodes then triangles ---
    snprintf(fnbuf, sizeof(fnbuf), "%s.bvh", base);
    f = fopen(fnbuf, "rb");
    if (!f) { perror(fnbuf); return 1; }

    unsigned int nNodes = 0;
    READ(nNodes, 1);
    srt::LinearBVHNode *nodes = new srt::LinearBVHNode[nNodes];
    for (unsigned int i = 0; i < nNodes; ++i) {
        float b[6];
        READ(b[0], 6);
        nodes[i].bounds[0][0] = b[0]; nodes[i].bounds[0][1] = b[1]; nodes[i].bounds[0][2] = b[2];
        nodes[i].bounds[1][0] = b[3]; nodes[i].bounds[1][1] = b[4]; nodes[i].bounds[1][2] = b[5];
        READ(nodes[i].offset, 1);
        READ(nodes[i].nPrimitives, 1);
        READ(nodes[i].splitAxis, 1);
        READ(nodes[i].pad, 1);
    }

    unsigned int nTris = 0;
    READ(nTris, 1);
    srt::Triangle *triangles = new srt::Triangle[nTris];
    for (unsigned int i = 0; i < nTris; ++i) {
        float v[9];
        READ(v[0], 9);
        float *vp = v;
        for (int j = 0; j < 3; ++j) {
            triangles[i].p[j][0] = *vp++;
            triangles[i].p[j][1] = *vp++;
            triangles[i].p[j][2] = *vp++;
            triangles[i].p[j][3] = 0.f;
        }
        triangles[i].id = i + 1;
    }
    fclose(f);

    int width = baseWidth, height = baseHeight; // scale = 1.0
    const int npix = width * height;
    printf("rt scene: %dx%d image, %u BVH nodes, %u triangles\n", width, height, nNodes, nTris);

    float *image = new float[npix];
    int *id = new int[npix];

    // ISPC struct types are byte-identical to srt:: ones (see rt_ispc.h).
    const ispc::LinearBVHNode *inodes = reinterpret_cast<const ispc::LinearBVHNode *>(nodes);
    const ispc::Triangle *itris = reinterpret_cast<const ispc::Triangle *>(triangles);

    const int WARMUP = 3, REPS = 15;

    // ---- serial ----
    for (int r = 0; r < WARMUP; ++r)
        srt::raytrace_serial(width, height, baseWidth, baseHeight, &raster2camera[0][0], &camera2world[0][0],
                             image, id, nodes, triangles);
    double bestSerial = 1e30;
    for (int r = 0; r < REPS; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        srt::raytrace_serial(width, height, baseWidth, baseHeight, &raster2camera[0][0], &camera2world[0][0],
                             image, id, nodes, triangles);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < bestSerial) bestSerial = ms;
    }
    double sumSerial = checksum_image(image, npix);

    // Save serial image as the canonical reference before ISPC overwrites it.
    if (dumpPath) {
        FILE *o = fopen(dumpPath, "wb");
        if (!o) { perror(dumpPath); return 1; }
        fwrite(image, sizeof(float), npix, o);
        fclose(o);
        fprintf(stderr, "wrote reference %s (%d floats)\n", dumpPath, npix);
    }

    // ---- ISPC ----
    for (int r = 0; r < WARMUP; ++r)
        raytrace_ispc(width, height, baseWidth, baseHeight, raster2camera, camera2world, image, id, inodes, itris);
    double bestIspc = 1e30;
    for (int r = 0; r < REPS; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        raytrace_ispc(width, height, baseWidth, baseHeight, raster2camera, camera2world, image, id, inodes, itris);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < bestIspc) bestIspc = ms;
    }
    double sumIspc = checksum_image(image, npix);

    printf("CHECKSUM_SERIAL %.10e\n", sumSerial);
    printf("CHECKSUM_ISPC   %.10e\n", sumIspc);
    printf("MS_SERIAL %.3f\n", bestSerial);
    printf("MS_ISPC   %.3f\n", bestIspc);
    printf("rel_checksum_diff %.3e\n", (sumSerial != 0.0) ? (sumIspc - sumSerial) / sumSerial : 0.0);

    delete[] nodes; delete[] triangles; delete[] image; delete[] id;
    return 0;
}
