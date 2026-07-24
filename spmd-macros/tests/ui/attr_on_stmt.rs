#![feature(portable_simd)]
#![allow(unused)]
use spmd::kernel;
use spmd::prelude::*;

#[kernel]
fn bad(x: Varying<i32>) -> Varying<i32> {
    let mut r = x;
    #[allow(clippy::all)]
    if x > 0 {
        r += 1;
    }
    r
}

fn main() {}
