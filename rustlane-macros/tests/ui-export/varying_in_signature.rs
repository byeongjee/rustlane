#![feature(portable_simd)]
#![allow(unused)]
use rustlane::export;
use rustlane::prelude::*;

#[export]
fn bad(x: Varying<f32>, out: &mut [f32]) {
    let _ = x;
    let _ = out;
}

fn main() {}
