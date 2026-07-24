#![feature(portable_simd)]
#![allow(unused)]
use rustlane::kernel;
use rustlane::prelude::*;

#[kernel]
fn bad(x: Varying<i32>) -> Varying<i32> {
    let mut acc = Varying::splat(0);
    helpers::stir(&mut acc);
    acc + x
}

fn main() {}
