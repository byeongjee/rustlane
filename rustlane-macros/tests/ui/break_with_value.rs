#![feature(portable_simd)]
#![allow(unused)]
use rustlane::kernel;
use rustlane::prelude::*;

#[kernel]
fn bad(n: Varying<i32>) -> Varying<i32> {
    let mut c = Varying::splat(0);
    loop {
        c += 1;
        if c >= n {
            break c;
        }
    }
    c
}

fn main() {}
