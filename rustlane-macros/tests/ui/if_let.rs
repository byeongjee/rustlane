#![feature(portable_simd)]
#![allow(unused)]
use rustlane::kernel;
use rustlane::prelude::*;

#[kernel]
fn bad(x: Varying<i32>, o: Option<i32>) -> Varying<i32> {
    let mut r = x;
    if let Some(v) = o {
        r += v;
    }
    r
}

fn main() {}
