#![feature(portable_simd)]
#![allow(unused)]
use rustlane::kernel;
use rustlane::prelude::*;

struct P {
    a: i32,
}

#[kernel]
fn bad(idx: Varying<i32>, out: &mut [P]) {
    out[idx].a = 1;
}

fn main() {}
