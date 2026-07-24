#![feature(portable_simd)]

use rustlane::prelude::*;
use rustlane::{export, kernel};
use std::time::Instant;

const LANES: usize = NATIVE_LANES;

#[kernel]
fn mandel(c_re: Varying<f32>, c_im: Varying<f32>, count: i32) -> Varying<i32> {
    let mut z_re = c_re;
    let mut z_im = c_im;
    let mut ret = Varying::splat(0);
    for i in 0..count {
        if z_re * z_re + z_im * z_im > 4.0 {
            break;
        }
        let new_re = z_re * z_re - z_im * z_im;
        let new_im = 2.0 * z_re * z_im;
        unmasked! {
            z_re = c_re + new_re;
            z_im = c_im + new_im;
        }
        ret = i + 1;
    }
    ret
}

#[export]
fn mandelbrot_render(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    width: i32,
    height: i32,
    max_iter: i32,
    output: &mut [i32],
) {
    let dx = (x1 - x0) / width as f32;
    let dy = (y1 - y0) / height as f32;
    let w = width as usize;
    let h = height as usize;
    foreach_2d!(y in 0..h, x in 0..w {
        let xf = x0 + (x.to_varying() as f32) * dx;
        let yf = Varying::<f32>::splat(y0 + (y as f32) * dy);
        let row = (y as i32) * width;
        output[x + row] = mandel(xf, yf, max_iter);
    });
}

fn mandelbrot_frame(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    width: i32,
    height: i32,
    max_iter: i32,
    output: &mut [i32],
) {
    let dx = (x1 - x0) / width as f32;
    let dy = (y1 - y0) / height as f32;
    let w = width as usize;
    let iota: Varying<f32, LANES> = Varying::from_array(std::array::from_fn(|l| l as f32));
    for j in 0..height {
        let y = Varying::<f32, LANES>::splat(y0 + j as f32 * dy);
        let mut base = 0usize;
        while base + LANES <= w {
            let x = Varying::splat(x0) + (Varying::splat(base as f32) + iota) * Varying::splat(dx);
            let r = mandel::<LANES, _>(AllOn, x, y, max_iter);
            let index = j as usize * w + base;
            r.0.copy_to_slice(&mut output[index..index + LANES]);
            base += LANES;
        }
    }
}

fn checksum(buf: &[i32]) -> i64 {
    buf.iter().map(|&v| v as i64).sum()
}

fn main() {
    let (width, height, max_iter, reps) = (768i32, 512i32, 256, 20);
    let n = (width * height) as usize;

    let mut buf_exp = vec![0i32; n];
    let mut best_exp = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        mandelbrot_render(-2.0, -1.0, 1.0, 1.0, width, height, max_iter, &mut buf_exp);
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        if dt < best_exp {
            best_exp = dt;
        }
    }

    let mut buf_ref = vec![0i32; n];
    let mut best_ref = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        mandelbrot_frame(-2.0, -1.0, 1.0, 1.0, width, height, max_iter, &mut buf_ref);
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        if dt < best_ref {
            best_ref = dt;
        }
    }

    let sum_exp = checksum(&buf_exp);
    let sum_ref = checksum(&buf_ref);
    println!(
        "mandelbrot(export,N={}):  {:.3} ms  (checksum {})",
        LANES, best_exp, sum_exp
    );
    println!(
        "mandelbrot(direct,N={}):  {:.3} ms  (checksum {})",
        LANES, best_ref, sum_ref
    );
    assert_eq!(sum_exp, 27304085, "exported checksum mismatch");
    assert_eq!(sum_exp, sum_ref, "exported vs direct checksum mismatch");
    println!("mandelbrot: OK (checksum 27304085, export == direct)");
}
