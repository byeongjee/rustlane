#![feature(portable_simd)]
#![allow(unused)]
use rustlane::kernel;
use rustlane::prelude::*;

#[kernel]
fn bad(x: Varying<i32>) -> Varying<i32> {
    let mut acc = Varying::splat(0);
    for (a, b) in [(1, 2), (3, 4)] {
        acc += x + a + b;
    }
    acc
}

fn main() {}
