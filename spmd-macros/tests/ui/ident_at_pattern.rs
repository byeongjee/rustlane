#![feature(portable_simd)]
#![allow(unused)]
use spmd::kernel;
use spmd::prelude::*;

#[kernel]
fn bad(x: Varying<i32>) -> Varying<i32> {
    let whole @ inner = x + 1;
    whole
}

fn main() {}
