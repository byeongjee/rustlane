#![feature(portable_simd)]
#![allow(unused)]
use spmd::kernel;
use spmd::prelude::*;

#[kernel]
fn bad<T>(x: Varying<i32>) -> Varying<i32> {
    x
}

fn main() {}
