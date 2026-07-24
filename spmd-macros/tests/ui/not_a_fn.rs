#![feature(portable_simd)]
#![allow(unused)]
use spmd::kernel;

#[kernel]
struct Bad {
    x: i32,
}

fn main() {}
