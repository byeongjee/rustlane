#![feature(portable_simd)]

use spmd::prelude::*;
use spmd::{export, kernel};
use std::simd::Mask;
use std::time::Instant;


#[kernel]
fn inside(px: Varying<f32>, py: Varying<f32>, pz: Varying<f32>) -> Mask<i32, N> {
    px >= 0.3 && px <= 1.8 && py >= -0.2 && py <= 2.3 && pz >= 0.3 && pz <= 1.8
}

#[kernel]
fn lerp(t: Varying<f32>, a: Varying<f32>, b: Varying<f32>) -> Varying<f32> {
    (1.0 - t) * a + t * b
}

#[kernel]
fn d_lookup(
    x: Varying<i32>,
    y: Varying<i32>,
    z: Varying<i32>,
    density: &[f32],
    nx: i32,
    ny: i32,
    nz: i32,
) -> Varying<f32> {
    let xc = math::clamp(x, Varying::splat(0), Varying::splat(nx - 1));
    let yc = math::clamp(y, Varying::splat(0), Varying::splat(ny - 1));
    let zc = math::clamp(z, Varying::splat(0), Varying::splat(nz - 1));
    let idx = zc * nx * ny + yc * nx + xc;
    density[idx]
}

#[kernel]
fn density_at(
    px: Varying<f32>,
    py: Varying<f32>,
    pz: Varying<f32>,
    density: &[f32],
    nx: i32,
    ny: i32,
    nz: i32,
) -> Varying<f32> {
    let voxx = (px - 0.3) / (1.8 - 0.3) * (nx as f32) - 0.5;
    let voxy = (py - (-0.2)) / (2.3 - (-0.2)) * (ny as f32) - 0.5;
    let voxz = (pz - 0.3) / (1.8 - 0.3) * (nz as f32) - 0.5;
    let vx = voxx as i32;
    let vy = voxy as i32;
    let vz = voxz as i32;
    let dfx = voxx - (vx as f32);
    let dfy = voxy - (vy as f32);
    let dfz = voxz - (vz as f32);
    let d00 = lerp(
        dfx,
        d_lookup(vx, vy, vz, density, nx, ny, nz),
        d_lookup(vx + 1, vy, vz, density, nx, ny, nz),
    );
    let d10 = lerp(
        dfx,
        d_lookup(vx, vy + 1, vz, density, nx, ny, nz),
        d_lookup(vx + 1, vy + 1, vz, density, nx, ny, nz),
    );
    let d01 = lerp(
        dfx,
        d_lookup(vx, vy, vz + 1, density, nx, ny, nz),
        d_lookup(vx + 1, vy, vz + 1, density, nx, ny, nz),
    );
    let d11 = lerp(
        dfx,
        d_lookup(vx, vy + 1, vz + 1, density, nx, ny, nz),
        d_lookup(vx + 1, vy + 1, vz + 1, density, nx, ny, nz),
    );
    let d0 = lerp(dfy, d00, d10);
    let d1 = lerp(dfy, d01, d11);
    let tri = lerp(dfz, d0, d1);
    let ins = inside(px, py, pz);
    tri.select(ins, Varying::splat(0.0))
}

#[kernel]
fn intersect_p(
    ox: Varying<f32>,
    oy: Varying<f32>,
    oz: Varying<f32>,
    dx: Varying<f32>,
    dy: Varying<f32>,
    dz: Varying<f32>,
    t0: &mut Varying<f32>,
    t1: &mut Varying<f32>,
) {
    let tnx = (0.3 - ox) / dx;
    let tfx = (1.8 - ox) / dx;
    let tny = (-0.2 - oy) / dy;
    let tfy = (2.3 - oy) / dy;
    let tnz = (0.3 - oz) / dz;
    let tfz = (1.8 - oz) / dz;
    let t0x = math::min(tnx, tfx);
    let t1x = math::max(tnx, tfx);
    let t0y = math::min(tny, tfy);
    let t1y = math::max(tny, tfy);
    let t0z = math::min(tnz, tfz);
    let t1z = math::max(tnz, tfz);
    *t0 = math::max(math::max(t0x, t0y), t0z);
    *t1 = math::min(math::min(t1x, t1y), t1z);
}

#[kernel]
fn distance_sq(
    ax: f32,
    ay: f32,
    az: f32,
    bx: Varying<f32>,
    by: Varying<f32>,
    bz: Varying<f32>,
) -> Varying<f32> {
    let ex = ax - bx;
    let ey = ay - by;
    let ez = az - bz;
    ex * ex + ey * ey + ez * ez
}

#[kernel]
fn transmittance(
    p0x: f32,
    p0y: f32,
    p0z: f32,
    p1x: Varying<f32>,
    p1y: Varying<f32>,
    p1z: Varying<f32>,
    sigma_t: f32,
    density: &[f32],
    nx: i32,
    ny: i32,
    nz: i32,
) -> Varying<f32> {
    let ddx = p0x - p1x;
    let ddy = p0y - p1y;
    let ddz = p0z - p1z;
    let mut rt0 = Varying::splat(0.0);
    let mut rt1 = Varying::splat(0.0);
    intersect_p(p1x, p1y, p1z, ddx, ddy, ddz, &mut rt0, &mut rt1);
    let ray_t0 = math::max(rt0, Varying::splat(0.0));
    let ray_t1 = rt1;
    let mut tau = Varying::splat(0.0);
    let raylen = math::sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
    let step_t = 0.2 / raylen;
    let mut t = ray_t0;
    let mut posx = p1x + ddx * ray_t0;
    let mut posy = p1y + ddy * ray_t0;
    let mut posz = p1z + ddz * ray_t0;
    let step_x = ddx * step_t;
    let step_y = ddy * step_t;
    let step_z = ddz * step_t;
    while t < ray_t1 {
        let mut dens = Varying::splat(0.0);
        unmasked! {
            dens = density_at(posx, posy, posz, density, nx, ny, nz);
        }
        tau += 0.2 * sigma_t * dens; 
        unmasked! {
            posx += step_x;
            posy += step_y;
            posz += step_z;
            t += step_t;
        }
    }
    math::exp(-tau)
}

#[kernel]
fn raymarch(
    ox: f32,
    oy: f32,
    oz: f32,
    dx: Varying<f32>,
    dy: Varying<f32>,
    dz: Varying<f32>,
    density: &[f32],
    nx: i32,
    ny: i32,
    nz: i32,
) -> Varying<f32> {
    let mut rt0 = Varying::splat(0.0);
    let mut rt1 = Varying::splat(0.0);
    intersect_p(
        Varying::splat(ox),
        Varying::splat(oy),
        Varying::splat(oz),
        dx,
        dy,
        dz,
        &mut rt0,
        &mut rt1,
    );
    let ray_t0 = math::max(rt0, Varying::splat(0.0));
    let ray_t1 = rt1;
    let mut tau = Varying::splat(0.0);
    let mut rad = Varying::splat(0.0);
    let raylen = math::sqrt(dx * dx + dy * dy + dz * dz);
    let step_t = 0.025 / raylen;
    let mut t = ray_t0;
    let mut posx = ox + dx * ray_t0;
    let mut posy = oy + dy * ray_t0;
    let mut posz = oz + dz * ray_t0;
    let step_x = dx * step_t;
    let step_y = dy * step_t;
    let step_z = dz * step_t;
    while t < ray_t1 {
        let mut dens = Varying::splat(0.0);
        unmasked! {
            dens = density_at(posx, posy, posz, density, nx, ny, nz);
        }
        let atten = math::exp(-tau);
        if atten < 0.005 {
            break;
        }
        let li = 40.0 / distance_sq(-1.0, 4.0, 1.5, posx, posy, posz)
            * transmittance(-1.0, 4.0, 1.5, posx, posy, posz, 20.0, density, nx, ny, nz);
        rad += 0.025 * atten * dens * 10.0 * (li + 0.25); 
        unmasked! {
            tau += 0.025 * 20.0 * dens;
            posx += step_x;
            posy += step_y;
            posz += step_z;
            t += step_t;
        }
    }
    let positive = rad > 0.0;
    let result = math::pow(rad, Varying::splat(1.0 / 2.2));
    result.select(positive, Varying::splat(0.0))
}

#[kernel]
fn generate_ray(
    x: Varying<f32>,
    y: Varying<f32>,
    r2c: &[f32],
    c2w: &[f32],
    dirx: &mut Varying<f32>,
    diry: &mut Varying<f32>,
    dirz: &mut Varying<f32>,
) {
    let camw = r2c[15usize];
    let camx = (r2c[0usize] * x + r2c[1usize] * y + r2c[3usize]) / camw;
    let camy = (r2c[4usize] * x + r2c[5usize] * y + r2c[7usize]) / camw;
    let camz = r2c[11usize] / camw;
    *dirx = c2w[0usize] * camx + c2w[1usize] * camy + c2w[2usize] * camz;
    *diry = c2w[4usize] * camx + c2w[5usize] * camy + c2w[6usize] * camz;
    *dirz = c2w[8usize] * camx + c2w[9usize] * camy + c2w[10usize] * camz;
}

#[export]
fn volume_render(
    density: &[f32],
    nx: i32,
    ny: i32,
    nz: i32,
    raster2camera: &[f32],
    camera2world: &[f32],
    width: i32,
    height: i32,
    image: &mut [f32],
) {
    let c33 = camera2world[15usize];
    let ox = camera2world[3usize] / c33;
    let oy = camera2world[7usize] / c33;
    let oz = camera2world[11usize] / c33;
    let w = width as usize;
    let h = height as usize;
    foreach_2d!(row in 0..h, col in 0..w {
        let xf = col.to_varying() as f32;
        let yf = Varying::<f32>::splat(row as f32);
        let mut dirx = Varying::<f32>::splat(0.0);
        let mut diry = Varying::<f32>::splat(0.0);
        let mut dirz = Varying::<f32>::splat(0.0);
        generate_ray(xf, yf, raster2camera, camera2world, &mut dirx, &mut diry, &mut dirz);
        let rowoff = (row as i32) * width;
        image[col + rowoff] = raymarch(ox, oy, oz, dirx, diry, dirz, density, nx, ny, nz);
    });
}


const CAMERA_PATH: &str = "/Users/byeongjee/side/rust-ispc/ispc-bench/camera.dat";
const VOLUME_PATH: &str = "/Users/byeongjee/side/rust-ispc/ispc-bench/density_lowres.vol";
const REF_PATH: &str = "/Users/byeongjee/side/rust-ispc/ispc-ref/ref-out/volume.bin";

fn load_camera() -> (usize, usize, Vec<f32>, Vec<f32>) {
    let text = std::fs::read_to_string(CAMERA_PATH).expect("read camera.dat");
    let mut it = text.split_whitespace();
    let width: usize = it.next().unwrap().parse().unwrap();
    let height: usize = it.next().unwrap().parse().unwrap();
    let r2c: Vec<f32> = (0..16).map(|_| it.next().unwrap().parse().unwrap()).collect();
    let c2w: Vec<f32> = (0..16).map(|_| it.next().unwrap().parse().unwrap()).collect();
    (width, height, r2c, c2w)
}

fn load_volume() -> (i32, i32, i32, Vec<f32>) {
    let text = std::fs::read_to_string(VOLUME_PATH).expect("read density_lowres.vol");
    let mut it = text.split_whitespace();
    let nx: i32 = it.next().unwrap().parse().unwrap();
    let ny: i32 = it.next().unwrap().parse().unwrap();
    let nz: i32 = it.next().unwrap().parse().unwrap();
    let count = (nx * ny * nz) as usize;
    let density: Vec<f32> = (0..count).map(|_| it.next().unwrap().parse().unwrap()).collect();
    (nx, ny, nz, density)
}

fn load_ref(npix: usize) -> Vec<f32> {
    let bytes = std::fs::read(REF_PATH).expect("read volume.bin");
    assert_eq!(bytes.len(), npix * 4, "ref size mismatch");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    let (width, height, r2c, c2w) = load_camera();
    let (nx, ny, nz, density) = load_volume();
    let npix = width * height;
    let mut image = vec![0f32; npix];

    let call = |img: &mut [f32]| {
        volume_render(
            &density, nx, ny, nz, &r2c, &c2w, width as i32, height as i32, img,
        );
    };

    for _ in 0..3 {
        call(&mut image);
    }
    let mut best = f64::INFINITY;
    for _ in 0..15 {
        let t0 = Instant::now();
        call(&mut image);
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        if dt < best {
            best = dt;
        }
    }

    let checksum: f64 = image.iter().map(|&v| v as f64).sum();

    let refimg = load_ref(npix);
    let ref_checksum: f64 = refimg.iter().map(|&v| v as f64).sum();
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut mism = 0usize;
    const REL_FLOOR: f64 = 1e-4;
    const REL_TOL: f64 = 1e-3;
    for (a, r) in image.iter().zip(refimg.iter()) {
        let av = *a as f64;
        let rv = *r as f64;
        let abs = (av - rv).abs();
        if abs > max_abs {
            max_abs = abs;
        }
        if rv.abs() > REL_FLOOR {
            let rel = abs / rv.abs();
            if rel > max_rel {
                max_rel = rel;
            }
            if rel > REL_TOL {
                mism += 1;
            }
        } else if abs > REL_TOL {
            mism += 1;
        }
    }
    let checksum_rel = (checksum - ref_checksum).abs() / ref_checksum.abs();

    println!("MS {:.3}", best);
    println!("CHECKSUM {:.6}", checksum);
    println!("workload {}x{} image, {}x{}x{} volume", width, height, nx, ny, nz);
    println!(
        "VALIDATION ref_checksum={:.6} checksum_rel={:.3e} max_rel={:.3e} max_abs={:.3e} mismatches={}/{}",
        ref_checksum, checksum_rel, max_rel, max_abs, mism, npix
    );

    let pass = checksum_rel < 1e-4 && (mism as f64) < (npix as f64) * 0.005;
    if pass {
        println!("volume: OK");
    } else {
        println!("volume: VALIDATION FAILED");
        std::process::exit(1);
    }
}
