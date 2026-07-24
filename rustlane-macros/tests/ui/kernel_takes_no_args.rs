#![feature(portable_simd)]
#![allow(unused)]
use rustlane::kernel;
use rustlane::prelude::*;

#[kernel(fast)]
fn bad(x: Varying<i32>) -> Varying<i32> {
    x
}

fn main() {}
