#![feature(portable_simd)]
#![allow(unused)]
use spmd::kernel;
use spmd::prelude::*;

#[kernel]
fn bad(x: Varying<i32>) -> Varying<i32> {
    let r = if x > 0 { x } else { -x };
    r
}

fn main() {}
