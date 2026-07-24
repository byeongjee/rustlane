#![feature(portable_simd)]
#![allow(unused)]
use rustlane::kernel;
use rustlane::prelude::*;

#[kernel]
fn bad(x: Varying<i32>) -> Varying<i32> {
    let (a, b) = (x, x + 1);
    a + b
}

fn main() {}
