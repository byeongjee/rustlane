#![feature(portable_simd)]
#![allow(unused)]
use spmd::kernel;
use spmd::prelude::*;

#[kernel]
fn bad(x: Varying<i32>, out: &mut [i32]) {
    if x > 0 {
        return 5;
    }
    out[0] = 1;
}

fn main() {}
