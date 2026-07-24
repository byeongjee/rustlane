#![feature(portable_simd)]
#![allow(unused)]
use rustlane::kernel;
use rustlane::prelude::*;

#[kernel]
fn bad(x: Varying<i32>, mut o: Option<i32>) -> Varying<i32> {
    let mut r = x;
    while let Some(v) = o {
        r += v;
        o = None;
    }
    r
}

fn main() {}
