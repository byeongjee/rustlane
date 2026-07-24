#![feature(portable_simd)]
#![allow(unused)]
use rustlane::kernel;
use rustlane::prelude::*;

#[kernel]
fn bad(self, x: Varying<i32>) -> Varying<i32> {
    x
}

fn main() {}
