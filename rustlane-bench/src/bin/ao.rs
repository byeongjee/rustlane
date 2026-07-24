#![feature(portable_simd)]

use rustlane::prelude::*;
use rustlane::{export, kernel, SpmdValue};
use std::time::Instant;

#[derive(SpmdValue, Clone, Copy)]
#[repr(C)]
#[allow(dead_code)] 
struct Isect {
    t: f32,
    px: f32,
    py: f32,
    pz: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    hit: i32,
}


#[kernel]
fn sphere_isect(
    isect: VaryingIsect<N>,
    ox: Varying<f32>,
    oy: Varying<f32>,
    oz: Varying<f32>,
    dx: Varying<f32>,
    dy: Varying<f32>,
    dz: Varying<f32>,
    cx: f32,
    cy: f32,
    cz: f32,
    r: f32,
) -> VaryingIsect<N> {
    let mut out = isect;
    let rsx = ox - cx;
    let rsy = oy - cy;
    let rsz = oz - cz;
    let b = rsx * dx + rsy * dy + rsz * dz;
    let c = rsx * rsx + rsy * rsy + rsz * rsz - r * r;
    let d = b * b - c;
    if d > 0.0 {
        let t = (0.0 - b) - math::sqrt(d);
        if t > 0.0 && t < isect.t {
            let hx = ox + dx * t;
            let hy = oy + dy * t;
            let hz = oz + dz * t;
            let nx = hx - cx;
            let ny = hy - cy;
            let nz = hz - cz;
            let ninv = math::rsqrt(nx * nx + ny * ny + nz * nz);
            out.t = t;
            out.hit = 1;
            out.px = hx;
            out.py = hy;
            out.pz = hz;
            out.nx = nx * ninv;
            out.ny = ny * ninv;
            out.nz = nz * ninv;
        }
    }
    out
}

#[kernel]
fn plane_isect(
    isect: VaryingIsect<N>,
    ox: Varying<f32>,
    oy: Varying<f32>,
    oz: Varying<f32>,
    dx: Varying<f32>,
    dy: Varying<f32>,
    dz: Varying<f32>,
) -> VaryingIsect<N> {
    let mut out = isect;
    let v = dy; 
    if math::abs(v) >= 1.0e-17 {
        let t = (0.0 - (oy + 0.5)) / v;
        if t > 0.0 && t < isect.t {
            out.t = t;
            out.hit = 1;
            out.px = ox + dx * t;
            out.py = oy + dy * t;
            out.pz = oz + dz * t;
            out.nx = 0.0;
            out.ny = 1.0;
            out.nz = 0.0;
        }
    }
    out
}


#[kernel]
fn ambient_occlusion(
    rng: &mut rng::RNGState<N>,
    ipx: Varying<f32>,
    ipy: Varying<f32>,
    ipz: Varying<f32>,
    inx: Varying<f32>,
    iny: Varying<f32>,
    inz: Varying<f32>,
) -> Varying<f32> {
    let eps = 0.0001;
    let px = ipx + eps * inx;
    let py = ipy + eps * iny;
    let pz = ipz + eps * inz;

    let b2x = inx;
    let b2y = iny;
    let b2z = inz;
    let mut b1x = Varying::splat(0.0);
    let mut b1y = Varying::splat(0.0);
    let mut b1z = Varying::splat(0.0);
    if math::abs(inx) < 0.6 {
        b1x = 1.0;
    } else {
        if math::abs(iny) < 0.6 {
            b1y = 1.0;
        } else {
            if math::abs(inz) < 0.6 {
                b1z = 1.0;
            } else {
                b1x = 1.0;
            }
        }
    }
    let mut b0x = b1y * b2z - b1z * b2y;
    let mut b0y = b1z * b2x - b1x * b2z;
    let mut b0z = b1x * b2y - b1y * b2x;
    let inv0 = math::rsqrt(b0x * b0x + b0y * b0y + b0z * b0z);
    b0x = b0x * inv0;
    b0y = b0y * inv0;
    b0z = b0z * inv0;
    b1x = b2y * b0z - b2z * b0y;
    b1y = b2z * b0x - b2x * b0z;
    b1z = b2x * b0y - b2y * b0x;
    let inv1 = math::rsqrt(b1x * b1x + b1y * b1y + b1z * b1z);
    b1x = b1x * inv1;
    b1y = b1y * inv1;
    b1z = b1z * inv1;

    let mut occ = Varying::splat(0);
    for _j in 0..8 {
        for _i in 0..8 {
            let theta = math::sqrt(rng.frandom());
            let phi = rng.frandom() * (2.0 * 3.1415926535);
            let sx = math::cos(phi) * theta;
            let sy = math::sin(phi) * theta;
            let sz = math::sqrt(1.0 - theta * theta);
            let rx = sx * b0x + sy * b1x + sz * b2x;
            let ry = sx * b0y + sy * b1y + sz * b2y;
            let rz = sx * b0z + sy * b1z + sz * b2z;
            let mut occ_isect = VaryingIsect {
                t: Varying::splat(1.0e17),
                px: Varying::splat(0.0),
                py: Varying::splat(0.0),
                pz: Varying::splat(0.0),
                nx: Varying::splat(0.0),
                ny: Varying::splat(0.0),
                nz: Varying::splat(0.0),
                hit: Varying::splat(0),
            };
            occ_isect = sphere_isect(occ_isect, px, py, pz, rx, ry, rz, -2.0, 0.0, -3.5, 0.5);
            occ_isect = sphere_isect(occ_isect, px, py, pz, rx, ry, rz, -0.5, 0.0, -3.0, 0.5);
            occ_isect = sphere_isect(occ_isect, px, py, pz, rx, ry, rz, 1.0, 0.0, -2.2, 0.5);
            occ_isect = plane_isect(occ_isect, px, py, pz, rx, ry, rz);
            occ += occ_isect.hit;
        }
    }
    let occf = occ as f32;
    (64.0 - occf) / 64.0
}

#[export]
fn ao_render(w: i32, h: i32, ns: i32, image: &mut [f32]) {
    let wf = w as f32;
    let hf = h as f32;
    let halfw = wf * 0.5;
    let halfh = hf * 0.5;
    let ratio = wf / hf;
    let inv_ss = 1.0 / (ns as f32);
    let inv_ss2 = inv_ss * inv_ss;

    let mut rngstate = rng::RNGState::<N>::new(reduce::lanes_iota::<N>() as u32);

    for y in 0..h {
        foreach!(x in 0..(w as usize) {
            let xi = x.to_varying();
            let xf = xi as f32;
            let mut acc = Varying::splat(0.0);
            for u in 0..ns {
                for v in 0..ns {
                    let du = (u as f32) * inv_ss;
                    let dv = (v as f32) * inv_ss;
                    let px_ndc = ((xf + du - halfw) / halfw) * ratio;
                    let py_ndc = Varying::splat(0.0 - (((y as f32) + dv - halfh) / halfh));
                    let inv = math::rsqrt(px_ndc * px_ndc + py_ndc * py_ndc + 1.0);
                    let dirx = px_ndc * inv;
                    let diry = py_ndc * inv;
                    let dirz = -1.0 * inv;
                    let orgx = Varying::splat(0.0);
                    let orgy = Varying::splat(0.0);
                    let orgz = Varying::splat(0.0);

                    let mut isect = VaryingIsect {
                        t: Varying::splat(1.0e17),
                        px: Varying::splat(0.0),
                        py: Varying::splat(0.0),
                        pz: Varying::splat(0.0),
                        nx: Varying::splat(0.0),
                        ny: Varying::splat(0.0),
                        nz: Varying::splat(0.0),
                        hit: Varying::splat(0),
                    };
                    isect = sphere_isect(isect, orgx, orgy, orgz, dirx, diry, dirz, -2.0, 0.0, -3.5, 0.5);
                    isect = sphere_isect(isect, orgx, orgy, orgz, dirx, diry, dirz, -0.5, 0.0, -3.0, 0.5);
                    isect = sphere_isect(isect, orgx, orgy, orgz, dirx, diry, dirz, 1.0, 0.0, -2.2, 0.5);
                    isect = plane_isect(isect, orgx, orgy, orgz, dirx, diry, dirz);

                    cif!(isect.hit > 0 => {
                        let o = ambient_occlusion(&mut rngstate,
                            isect.px, isect.py, isect.pz, isect.nx, isect.ny, isect.nz);
                        acc += o * inv_ss2;
                    });
                }
            }
            let base = 3 * ((y * w) + xi);
            image[base] = acc;
            image[base + 1] = acc;
            image[base + 2] = acc;
        });
    }
}


const REF_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../ispc-ref/ref-out/ao_ispc.bin");
const REF_CHECKSUM: f64 = 365161.535156;

fn load_ref(n: usize) -> Option<Vec<f32>> {
    let bytes = std::fs::read(REF_PATH).ok()?;
    if bytes.len() != n * 4 {
        return None;
    }
    let mut out = vec![0f32; n];
    for (i, c) in bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    Some(out)
}

fn main() {
    let (w, h, ns) = (512i32, 512i32, 2i32);
    let n = (w * h * 3) as usize;
    let (warm, reps) = (3, 15);

    let mut image = vec![0f32; n];
    let mut best = f64::INFINITY;
    for r in 0..(warm + reps) {
        image.iter_mut().for_each(|p| *p = 0.0);
        let t0 = Instant::now();
        ao_render(w, h, ns, &mut image);
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        if r >= warm && dt < best {
            best = dt;
        }
    }

    let checksum: f64 = image.iter().map(|&v| v as f64).sum();
    let (mut vmin, mut vmax) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in &image {
        vmin = vmin.min(v);
        vmax = vmax.max(v);
    }

    println!("MS ao {:.3}", best);
    println!(
        "CHECKSUM ours {:.6} ref {:.6}  (512x512 ns=2, N={}, range [{:.3},{:.3}])",
        checksum, REF_CHECKSUM, NATIVE_LANES, vmin, vmax
    );

    let mut ok = true;
    let mut detail = String::new();
    match load_ref(n) {
        Some(refimg) => {
            let ref_sum: f64 = refimg.iter().map(|&v| v as f64).sum();
            let mut sum_abs = 0.0f64;
            let mut max_abs = 0.0f64;
            for (a, b) in image.iter().zip(refimg.iter()) {
                let ad = (*a as f64 - *b as f64).abs();
                sum_abs += ad;
                if ad > max_abs {
                    max_abs = ad;
                }
            }
            let mean_abs = sum_abs / n as f64;
            let rel = (checksum - ref_sum).abs() / ref_sum.abs();
            detail = format!(
                "checksum rel err {:.4}% vs ref_sum {:.3} (documented {:.3}); per-pixel mean abs diff {:.5}, max abs diff {:.5}",
                rel * 100.0,
                ref_sum,
                REF_CHECKSUM,
                mean_abs,
                max_abs
            );
            println!("VALIDATION {}", detail);
            if rel > 0.001 {
                ok = false;
            }
        }
        None => {
            let rel = (checksum - REF_CHECKSUM).abs() / REF_CHECKSUM.abs();
            detail = format!(
                "ref file unavailable; checksum rel err {:.4}% vs documented {:.3}",
                rel * 100.0,
                REF_CHECKSUM
            );
            println!("VALIDATION {}", detail);
            if rel > 0.001 {
                ok = false;
            }
        }
    }

    if ok {
        println!("ao: OK ({})", detail);
    } else {
        eprintln!("ao: VALIDATION FAILED ({})", detail);
        std::process::exit(1);
    }
}
