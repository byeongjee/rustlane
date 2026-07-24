#![feature(portable_simd)]
#![allow(unused)]
use spmd::kernel;
use spmd::prelude::*;

#[kernel]
fn double(x: Varying<i32>) -> Varying<i32> {
    x * 2
}

#[kernel]
fn bad(x: Varying<i32>) -> Varying<i32> {
    double::<8>(x)
}

fn main() {}
