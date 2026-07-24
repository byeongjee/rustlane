#![feature(portable_simd)]
#![allow(unused)]
use rustlane::kernel;
use rustlane::prelude::*;

#[kernel]
fn bad(N: i32) -> Varying<i32> {
    Varying::splat(N)
}

fn main() {}
