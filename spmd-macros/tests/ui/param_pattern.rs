#![feature(portable_simd)]
#![allow(unused)]
use spmd::kernel;
use spmd::prelude::*;

#[kernel]
fn bad((a, b): (i32, i32)) -> Varying<i32> {
    Varying::splat(a + b)
}

fn main() {}
