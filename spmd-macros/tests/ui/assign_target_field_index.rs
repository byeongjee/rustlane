#![feature(portable_simd)]
#![allow(unused)]
use spmd::kernel;
use spmd::prelude::*;

struct P {
    a: i32,
}

#[kernel]
fn bad(idx: Varying<i32>, out: &mut [P]) {
    out[idx].a = 1;
}

fn main() {}
