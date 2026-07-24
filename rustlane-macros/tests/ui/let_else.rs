#![feature(portable_simd)]
#![allow(unused)]
use rustlane::kernel;
use rustlane::prelude::*;

#[kernel]
fn bad(x: Varying<i32>, o: i32) -> Varying<i32> {
    let v = o else {
        return x;
    };
    x + v
}

fn main() {}
