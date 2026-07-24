#![feature(portable_simd)]
#![allow(unused)]
use spmd::kernel;
use spmd::prelude::*;

#[kernel]
fn bad(x: Varying<i32>) -> Varying<i32> {
    unmasked! {
        return Varying::splat(7);
    }
    x
}

fn main() {}
