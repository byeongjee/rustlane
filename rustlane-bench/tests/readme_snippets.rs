#![feature(portable_simd)]

use rustlane::prelude::*;
use rustlane::{export, kernel};

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

#[kernel]
fn scale(x: Varying<f32>, factor: f32) -> Varying<f32> {
    x * factor
}

#[export]
fn scale_all(input: &[f32], output: &mut [f32], factor: f32) {
    foreach!(i in 0..input.len() {
        output[i] = scale(input[i], factor);
    });
}

#[test]
fn quick_start_runs() {
    let input: Vec<f32> = (0..1024).map(|v| v as f32).collect();
    let mut output = vec![0.0f32; input.len()];
    scale_all(&input, &mut output, 2.0);
    assert_eq!(output[10], 20.0);
    assert_eq!(output.len(), 1024);
}

#[test]
fn hero_kernel_instantiates() {
    let c_re: Varying<f32, 8> = Varying::splat(0.0);
    let c_im: Varying<f32, 8> = Varying::splat(0.0);
    let out = mandel::<8, _>(AllOn, c_re, c_im, 16);
    assert_eq!(out.0.to_array(), [16i32; 8]);
}
