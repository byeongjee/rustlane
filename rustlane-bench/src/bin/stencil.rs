use rustlane::export;
use std::time::Instant;

#[export]
fn stencil_step(
    x0: usize,
    x1: usize,
    y0: i32,
    y1: i32,
    z0: i32,
    z1: i32,
    nx: i32,
    nxy: i32,
    c0: f32,
    c1: f32,
    c2: f32,
    c3: f32,
    vsq: &[f32],
    ain: &[f32],
    aout: &mut [f32],
) {
    let s1y = nx;
    let s2y = nx * 2;
    let s3y = nx * 3;
    let s1z = nxy;
    let s2z = nxy * 2;
    let s3z = nxy * 3;
    for z in z0..z1 {
        for y in y0..y1 {
            let row = z * nxy + y * nx;
            foreach!(x in x0..x1 {
                let idx = x + row;
                let cur = ain[idx];
                let n1 = ain[idx + 1] + ain[idx + (-1)]
                    + ain[idx + s1y] + ain[idx + (-s1y)]
                    + ain[idx + s1z] + ain[idx + (-s1z)];
                let n2 = ain[idx + 2] + ain[idx + (-2)]
                    + ain[idx + s2y] + ain[idx + (-s2y)]
                    + ain[idx + s2z] + ain[idx + (-s2z)];
                let n3 = ain[idx + 3] + ain[idx + (-3)]
                    + ain[idx + s3y] + ain[idx + (-s3y)]
                    + ain[idx + s3z] + ain[idx + (-s3z)];
                let div = c0 * cur + c1 * n1 + c2 * n2 + c3 * n3;
                let prev = aout[idx];
                aout[idx] = 2.0 * cur - prev + vsq[idx] * div;
            });
        }
    }
}

fn init_data(nx: i32, ny: i32, nz: i32, a0: &mut [f32], a1: &mut [f32], vsq: &mut [f32]) {
    let denom = (nx * ny * nz) as f32;
    let mut off = 0usize;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                a0[off] = if x < nx / 2 {
                    x as f32 / nx as f32
                } else {
                    y as f32 / ny as f32
                };
                a1[off] = 0.0;
                vsq[off] = (x * y * z) as f32 / denom;
                off += 1;
            }
        }
    }
}

fn checksum(buf: &[f32]) -> f64 {
    buf.iter().map(|&v| v as f64).sum()
}

fn load_ref(path: &str, n: usize) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() != n * 4 {
        eprintln!(
            "# reference {path}: expected {} bytes, got {}",
            n * 4,
            bytes.len()
        );
        return None;
    }
    let mut out = vec![0.0f32; n];
    for (i, o) in out.iter_mut().enumerate() {
        *o = f32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ]);
    }
    Some(out)
}

fn main() {
    let (nx, ny, nz, w) = (256i32, 256i32, 256i32, 4i32);
    let nxy = nx * ny;
    let n = (nx as usize) * (ny as usize) * (nz as usize);
    let (t0, t1) = (0i32, 6i32);
    let (c0, c1, c2, c3) = (0.5f32, -0.25f32, 0.125f32, -0.0625f32);
    let (warmup, reps) = (3, 15);

    let (x0, x1) = (w as usize, (nx - w) as usize);
    let (y0, y1) = (w, ny - w);
    let (z0, z1) = (w, nz - w);

    let mut a0 = vec![0.0f32; n];
    let mut a1 = vec![0.0f32; n];
    let mut vsq = vec![0.0f32; n];

    let mut best = f64::INFINITY;
    for r in 0..(warmup + reps) {
        init_data(nx, ny, nz, &mut a0, &mut a1, &mut vsq);
        let start = Instant::now();
        for t in t0..t1 {
            if (t & 1) == 0 {
                stencil_step(
                    x0, x1, y0, y1, z0, z1, nx, nxy, c0, c1, c2, c3, &vsq, &a0, &mut a1,
                );
            } else {
                stencil_step(
                    x0, x1, y0, y1, z0, z1, nx, nxy, c0, c1, c2, c3, &vsq, &a1, &mut a0,
                );
            }
        }
        let ms = start.elapsed().as_secs_f64() * 1e3;
        if r >= warmup && ms < best {
            best = ms;
        }
    }

    let last_t = t1 - 1;
    let final_buf: &[f32] = if (last_t & 1) == 0 { &a1 } else { &a0 };
    let sum = checksum(final_buf);

    println!("MS stencil {best:.3}");
    println!("CHECKSUM {sum:.6}");

    let ref_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../ispc-ref/ref-out/stencil.bin"
    );
    let mut validated = false;
    if let Some(reference) = load_ref(ref_path, n) {
        let mut bit_exact = 0usize;
        let mut max_rel = 0.0f64;
        let mut max_abs = 0.0f64;
        for (i, &got) in final_buf.iter().enumerate() {
            let exp = reference[i];
            if got.to_bits() == exp.to_bits() {
                bit_exact += 1;
                continue;
            }
            let a = (got as f64 - exp as f64).abs();
            let rel = a / (exp as f64).abs().max(1e-30);
            if a > max_abs {
                max_abs = a;
            }
            if rel > max_rel {
                max_rel = rel;
            }
        }
        let ref_sum = checksum(&reference);
        let checksum_rel = ((sum - ref_sum) / ref_sum).abs();
        validated = checksum_rel < 1e-6 && max_abs < 1e-3;
        eprintln!(
            "# validate: bit_exact={}/{}, max_abs={:.3e}, max_rel={:.3e}, \
             checksum_rel={:.3e} (ref_checksum={:.6})",
            bit_exact, n, max_abs, max_rel, checksum_rel, ref_sum
        );
        if validated {
            println!(
                "stencil: OK (fma-tolerance vs stencil.bin; checksum_rel {checksum_rel:.2e}, max_abs {max_abs:.2e})"
            );
        } else {
            println!("stencil: FAIL (checksum_rel {checksum_rel:.2e}, max_abs {max_abs:.2e})");
        }
    } else {
        eprintln!("# reference {ref_path} not loaded; skipping validation");
    }

    if !validated {
        std::process::exit(1);
    }
}
