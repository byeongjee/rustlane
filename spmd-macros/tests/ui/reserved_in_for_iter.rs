#![feature(portable_simd)]
#![allow(unused)]
use spmd::kernel;
use spmd::prelude::*;

#[kernel]
fn bad(x: Varying<i32>) -> Varying<i32> {
    let mut acc = Varying::splat(0);
    for _ in {
        let __stolen = __exec;
        0..2
    } {
        acc += x;
    }
    acc
}

fn main() {}
