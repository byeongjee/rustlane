#![feature(portable_simd)]
#![allow(unused)]
use spmd::kernel;
use spmd::prelude::*;

#[kernel]
fn bad(x: Varying<i32>) -> Varying<i32> {
    let f = |v: Varying<i32>| v + 1;
    f(x)
}

fn main() {}
