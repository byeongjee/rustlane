#![feature(portable_simd)]
#![allow(unused)]
use rustlane::kernel;

#[kernel]
struct Bad {
    x: i32,
}

fn main() {}
