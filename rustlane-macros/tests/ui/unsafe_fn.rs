#![feature(portable_simd)]
#![allow(unused)]
use rustlane::kernel;
use rustlane::prelude::*;

#[kernel]
unsafe fn bad(x: Varying<i32>) -> Varying<i32> {
    x
}

fn main() {}
