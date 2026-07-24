#![feature(portable_simd)]
#![allow(unused)]
use spmd::kernel;
use spmd::prelude::*;

#[kernel]
fn bad(N: i32) -> Varying<i32> {
    Varying::splat(N)
}

fn main() {}
