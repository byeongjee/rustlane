#![feature(portable_simd)]
#![allow(unused)]
use spmd::kernel;
use spmd::prelude::*;

#[kernel]
fn bad(x: Varying<i32>, k: i32) -> Varying<i32> {
    let mut acc = x;
    while k < 4 {
        acc += k;
    }
    acc
}

fn main() {}
