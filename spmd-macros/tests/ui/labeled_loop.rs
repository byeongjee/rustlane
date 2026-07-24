#![feature(portable_simd)]
#![allow(unused)]
use spmd::kernel;
use spmd::prelude::*;

#[kernel]
fn bad(n: Varying<i32>) -> Varying<i32> {
    let mut c = Varying::splat(0);
    'outer: loop {
        c += 1;
        if c >= n {
            break 'outer;
        }
    }
    c
}

fn main() {}
