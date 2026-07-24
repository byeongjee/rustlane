#![feature(portable_simd)]
#![allow(unused)]
use rustlane::kernel;
use rustlane::prelude::*;

#[kernel]
fn bad(x: Varying<i32>) -> Varying<i32> {
    if x > 100 {
        return Varying::splat(-1);
    }
    __ret = x + 1000;
    __exec += 1;
    x
}

fn main() {}
