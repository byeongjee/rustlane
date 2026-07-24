#![feature(portable_simd)]

use spmd::export;
use std::time::Instant;


#[derive(Clone, Copy)]
struct BvhNode {
    bmin: [f32; 3],
    bmax: [f32; 3],
    offset: u32,       
    n_primitives: u8,  
    split_axis: u8,
}

#[derive(Clone, Copy)]
struct Triangle {
    p: [[f32; 3]; 3],
    id: i32,
}


mod scene {
    use super::{BvhNode, Triangle};
    use spmd::prelude::*;
    use std::simd::{LaneCount, Mask, StdFloat, SupportedLaneCount};

    #[inline(always)]
    fn vfma<const N: usize>(
        a: Varying<f32, N>,
        b: Varying<f32, N>,
        c: Varying<f32, N>,
    ) -> Varying<f32, N>
    where
        LaneCount<N>: SupportedLaneCount,
    {
        Varying(a.0.mul_add(b.0, c.0))
    }

    #[inline(always)]
    fn cross<const N: usize>(
        ax: Varying<f32, N>,
        ay: Varying<f32, N>,
        az: Varying<f32, N>,
        bx: Varying<f32, N>,
        by: Varying<f32, N>,
        bz: Varying<f32, N>,
    ) -> (Varying<f32, N>, Varying<f32, N>, Varying<f32, N>)
    where
        LaneCount<N>: SupportedLaneCount,
    {
        (
            vfma(ay, bz, -(az * by)),
            vfma(az, bx, -(ax * bz)),
            vfma(ax, by, -(ay * bx)),
        )
    }

    #[inline(always)]
    fn dot3<const N: usize>(
        ax: Varying<f32, N>,
        ay: Varying<f32, N>,
        az: Varying<f32, N>,
        bx: Varying<f32, N>,
        by: Varying<f32, N>,
        bz: Varying<f32, N>,
    ) -> Varying<f32, N>
    where
        LaneCount<N>: SupportedLaneCount,
    {
        vfma(az, bz, vfma(ax, bx, ay * by))
    }

    #[derive(Clone, Copy)]
    pub struct V3<const N: usize>
    where
        LaneCount<N>: SupportedLaneCount,
    {
        pub x: Varying<f32, N>,
        pub y: Varying<f32, N>,
        pub z: Varying<f32, N>,
    }

    #[derive(Clone, Copy)]
    pub struct Ray<const N: usize>
    where
        LaneCount<N>: SupportedLaneCount,
    {
        pub origin: V3<N>,
        pub dir: V3<N>,
        pub inv_dir: V3<N>,
        pub dir_is_neg: [bool; 3],
    }

    pub struct HitResult<const N: usize>
    where
        LaneCount<N>: SupportedLaneCount,
    {
        pub maxt: Varying<f32, N>,
        pub id: Varying<i32, N>,
    }

    #[inline(always)]
    pub fn generate_ray<const N: usize>(
        r2c: &[f32],
        c2w: &[f32],
        px: Varying<i32, N>,
        py: Varying<i32, N>,
        ws: f32,
        hs: f32,
    ) -> Ray<N>
    where
        LaneCount<N>: SupportedLaneCount,
    {
        let x = px.cast::<f32>() * ws;
        let y = py.cast::<f32>() * hs;

        let camx = vfma(Varying::splat(r2c[0]), x, Varying::splat(r2c[1]) * y) + r2c[3];
        let camy = vfma(Varying::splat(r2c[4]), x, Varying::splat(r2c[5]) * y) + r2c[7];
        let camz = r2c[11];
        let camw = r2c[15];
        let camx = camx / camw;
        let camy = camy / camw;
        let camz = camz / camw;
        let camz_v = Varying::splat(camz);

        let dirx = vfma(
            Varying::splat(c2w[2]),
            camz_v,
            vfma(Varying::splat(c2w[0]), camx, Varying::splat(c2w[1]) * camy),
        );
        let diry = vfma(
            Varying::splat(c2w[6]),
            camz_v,
            vfma(Varying::splat(c2w[4]), camx, Varying::splat(c2w[5]) * camy),
        );
        let dirz = vfma(
            Varying::splat(c2w[10]),
            camz_v,
            vfma(Varying::splat(c2w[8]), camx, Varying::splat(c2w[9]) * camy),
        );

        let ox = c2w[3] / c2w[15];
        let oy = c2w[7] / c2w[15];
        let oz = c2w[11] / c2w[15];

        let invx = 1.0f32 / dirx;
        let invy = 1.0f32 / diry;
        let invz = 1.0f32 / dirz;

        let dir_is_neg = [
            reduce::any(invx.spmd_lt(0.0f32)),
            reduce::any(invy.spmd_lt(0.0f32)),
            reduce::any(invz.spmd_lt(0.0f32)),
        ];

        Ray {
            origin: V3 {
                x: Varying::splat(ox),
                y: Varying::splat(oy),
                z: Varying::splat(oz),
            },
            dir: V3 { x: dirx, y: diry, z: dirz },
            inv_dir: V3 { x: invx, y: invy, z: invz },
            dir_is_neg,
        }
    }

    #[inline(always)]
    pub fn bbox_test<const N: usize>(
        node: &BvhNode,
        ray: &Ray<N>,
        mint: f32,
        maxt: Varying<f32, N>,
    ) -> Mask<i32, N>
    where
        LaneCount<N>: SupportedLaneCount,
    {
        let o = &ray.origin;
        let inv = &ray.inv_dir;

        let nx = (node.bmin[0] - o.x) * inv.x;
        let fx = (node.bmax[0] - o.x) * inv.x;
        let ny = (node.bmin[1] - o.y) * inv.y;
        let fy = (node.bmax[1] - o.y) * inv.y;
        let nz = (node.bmin[2] - o.z) * inv.z;
        let fz = (node.bmax[2] - o.z) * inv.z;

        let mut t0 = Varying::splat(mint);
        let mut t1 = maxt;
        t0 = math::max(t0, math::min(nx, fx));
        t1 = math::min(t1, math::max(nx, fx));
        t0 = math::max(t0, math::min(ny, fy));
        t1 = math::min(t1, math::max(ny, fy));
        t0 = math::max(t0, math::min(nz, fz));
        t1 = math::min(t1, math::max(nz, fz));
        t0.spmd_le(t1)
    }

    #[inline(always)]
    pub fn tri_test<const N: usize>(
        tri: &Triangle,
        ray: &Ray<N>,
        mint: f32,
        maxt: Varying<f32, N>,
    ) -> (Mask<i32, N>, Varying<f32, N>)
    where
        LaneCount<N>: SupportedLaneCount,
    {
        let p0x = Varying::splat(tri.p[0][0]);
        let p0y = Varying::splat(tri.p[0][1]);
        let p0z = Varying::splat(tri.p[0][2]);
        let e1x = Varying::splat(tri.p[1][0] - tri.p[0][0]);
        let e1y = Varying::splat(tri.p[1][1] - tri.p[0][1]);
        let e1z = Varying::splat(tri.p[1][2] - tri.p[0][2]);
        let e2x = Varying::splat(tri.p[2][0] - tri.p[0][0]);
        let e2y = Varying::splat(tri.p[2][1] - tri.p[0][1]);
        let e2z = Varying::splat(tri.p[2][2] - tri.p[0][2]);

        let dirx = ray.dir.x;
        let diry = ray.dir.y;
        let dirz = ray.dir.z;

        let (s1x, s1y, s1z) = cross(dirx, diry, dirz, e2x, e2y, e2z);
        let divisor = dot3(s1x, s1y, s1z, e1x, e1y, e1z);
        let inv_div = 1.0f32 / divisor;

        let dx = ray.origin.x - p0x;
        let dy = ray.origin.y - p0y;
        let dz = ray.origin.z - p0z;

        let b1 = dot3(dx, dy, dz, s1x, s1y, s1z) * inv_div;

        let (s2x, s2y, s2z) = cross(dx, dy, dz, e1x, e1y, e1z);

        let b2 = dot3(dirx, diry, dirz, s2x, s2y, s2z) * inv_div;
        let t = dot3(e2x, e2y, e2z, s2x, s2y, s2z) * inv_div;

        let hit = divisor.spmd_ne(0.0f32)
            & b1.spmd_ge(0.0f32)
            & b1.spmd_le(1.0f32)
            & b2.spmd_ge(0.0f32)
            & (b1 + b2).spmd_le(1.0f32)
            & t.spmd_ge(mint)
            & t.spmd_le(maxt);
        (hit, t)
    }

    #[inline(always)]
    pub fn bvh_intersect<const N: usize>(
        nodes: &[BvhNode],
        tris: &[Triangle],
        ray: &Ray<N>,
    ) -> HitResult<N>
    where
        LaneCount<N>: SupportedLaneCount,
    {
        let mint = 0.0f32;
        let mut maxt = Varying::splat(1e30f32);
        let mut hit_id = Varying::splat(0i32);

        let mut node_num = 0usize;
        let mut todo = [0usize; 64];
        let mut todo_off = 0usize;

        loop {
            let node = nodes[node_num];
            if reduce::any(bbox_test(&node, ray, mint, maxt)) {
                if node.n_primitives > 0 {
                    let poff = node.offset as usize;
                    for i in 0..node.n_primitives as usize {
                        let tri = &tris[poff + i];
                        let (hit, t) = tri_test(tri, ray, mint, maxt);
                        maxt = t.select(hit, maxt);
                        hit_id = Varying::splat(tri.id).select(hit, hit_id);
                    }
                    if todo_off == 0 {
                        break;
                    }
                    todo_off -= 1;
                    node_num = todo[todo_off];
                } else {
                    if ray.dir_is_neg[node.split_axis as usize] {
                        todo[todo_off] = node_num + 1;
                        node_num = node.offset as usize;
                    } else {
                        todo[todo_off] = node.offset as usize;
                        node_num = node_num + 1;
                    }
                    todo_off += 1;
                }
            } else {
                if todo_off == 0 {
                    break;
                }
                todo_off -= 1;
                node_num = todo[todo_off];
            }
        }

        HitResult { maxt, id: hit_id }
    }

    #[inline(always)]
    pub fn render_pixel<const N: usize>(
        r2c: &[f32],
        c2w: &[f32],
        px: Varying<i32, N>,
        py: Varying<i32, N>,
        ws: f32,
        hs: f32,
        nodes: &[BvhNode],
        tris: &[Triangle],
    ) -> HitResult<N>
    where
        LaneCount<N>: SupportedLaneCount,
    {
        let ray = generate_ray(r2c, c2w, px, py, ws, hs);
        bvh_intersect(nodes, tris, &ray)
    }
}


#[export]
fn raytrace(
    width: i32,
    height: i32,
    base_width: i32,
    base_height: i32,
    r2c: &[f32],
    c2w: &[f32],
    image: &mut [f32],
    id: &mut [i32],
    nodes: &[BvhNode],
    tris: &[Triangle],
) {
    let ws = base_width as f32 / width as f32;
    let hs = base_height as f32 / height as f32;
    let npix = (width * height) as usize;
    foreach!(p in 0..npix {
        let idx = p.to_varying();
        let py = idx / width;
        let px = idx - py * width;
        let hit = scene::render_pixel(r2c, c2w, px, py, ws, hs, nodes, tris);
        image[p] = hit.maxt;
        id[p] = hit.id;
    });
}


const SCENE_BASE: &str = "/Users/byeongjee/side/rust-ispc/ispc-bench/sponza";
const REF_PATH: &str = "/Users/byeongjee/side/rust-ispc/ispc-ref/ref-out/rt.bin";

fn rd_f32(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}
fn rd_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}
fn rd_i32(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn load_camera() -> (i32, i32, [f32; 16], [f32; 16]) {
    let b = std::fs::read(format!("{SCENE_BASE}.camera")).expect("read camera");
    let base_width = rd_i32(&b, 0);
    let base_height = rd_i32(&b, 4);
    let mut c2w = [0f32; 16]; 
    let mut r2c = [0f32; 16]; 
    for i in 0..16 {
        c2w[i] = rd_f32(&b, 8 + i * 4);
    }
    for i in 0..16 {
        r2c[i] = rd_f32(&b, 8 + 64 + i * 4);
    }
    (base_width, base_height, r2c, c2w)
}

fn load_bvh() -> (Vec<BvhNode>, Vec<Triangle>) {
    let b = std::fs::read(format!("{SCENE_BASE}.bvh")).expect("read bvh");
    let mut off = 0usize;
    let n_nodes = rd_u32(&b, off) as usize;
    off += 4;
    let mut nodes = Vec::with_capacity(n_nodes);
    for _ in 0..n_nodes {
        let bmin = [rd_f32(&b, off), rd_f32(&b, off + 4), rd_f32(&b, off + 8)];
        let bmax = [rd_f32(&b, off + 12), rd_f32(&b, off + 16), rd_f32(&b, off + 20)];
        let offset = rd_u32(&b, off + 24);
        let n_primitives = b[off + 28];
        let split_axis = b[off + 29];
        nodes.push(BvhNode { bmin, bmax, offset, n_primitives, split_axis });
        off += 32;
    }
    let n_tris = rd_u32(&b, off) as usize;
    off += 4;
    let mut tris = Vec::with_capacity(n_tris);
    for i in 0..n_tris {
        let mut p = [[0f32; 3]; 3];
        let mut vp = off;
        for j in 0..3 {
            p[j][0] = rd_f32(&b, vp);
            p[j][1] = rd_f32(&b, vp + 4);
            p[j][2] = rd_f32(&b, vp + 8);
            vp += 12;
        }
        tris.push(Triangle { p, id: (i + 1) as i32 });
        off += 36;
    }
    (nodes, tris)
}

fn main() {
    let (base_width, base_height, r2c, c2w) = load_camera();
    let (nodes, tris) = load_bvh();
    let width = base_width; 
    let height = base_height;
    let npix = (width * height) as usize;
    println!(
        "rt scene: {}x{} image, {} BVH nodes, {} triangles",
        width,
        height,
        nodes.len(),
        tris.len()
    );

    let mut image = vec![0f32; npix];
    let mut id = vec![0i32; npix];

    let (warmup, reps) = (3usize, 15usize);
    for _ in 0..warmup {
        raytrace(width, height, base_width, base_height, &r2c, &c2w, &mut image, &mut id, &nodes, &tris);
    }
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        raytrace(width, height, base_width, base_height, &r2c, &c2w, &mut image, &mut id, &nodes, &tris);
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        if dt < best {
            best = dt;
        }
    }

    let checksum: f64 = image.iter().map(|&v| v as f64).sum();

    let refbytes = std::fs::read(REF_PATH).expect("read rt.bin");
    assert_eq!(refbytes.len(), npix * 4, "ref size mismatch");
    let mut max_rel = 0.0f64;
    let mut mismatches = 0usize;
    for i in 0..npix {
        let r = rd_f32(&refbytes, i * 4) as f64;
        let m = image[i] as f64;
        let denom = r.abs().max(1e-30);
        let rel = (m - r).abs() / denom;
        if rel > max_rel {
            max_rel = rel;
        }
        if rel > 1e-6 {
            mismatches += 1;
        }
    }
    let refsum: f64 = (0..npix).map(|i| rd_f32(&refbytes, i * 4) as f64).sum();
    let checksum_rel = ((checksum - refsum) / refsum).abs();

    println!("CHECKSUM {:.10e}", checksum);
    println!("MS {:.3}", best);
    println!(
        "validation: max_rel_err {:.3e}, mismatches {} / {}, checksum_rel {:.3e}",
        max_rel, mismatches, npix, checksum_rel
    );

    let ok = max_rel <= 1e-6 && checksum_rel <= 1e-6;
    if ok {
        println!("rt: OK");
    } else {
        eprintln!("rt: VALIDATION FAILED");
        std::process::exit(1);
    }
}
