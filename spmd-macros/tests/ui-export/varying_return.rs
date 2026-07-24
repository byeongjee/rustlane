#![feature(portable_simd)]
#![allow(unused)]
use spmd::export;
use spmd::prelude::*;

#[export]
fn bad(n: i32) -> Varying<i32> {
    Varying::splat(n)
}

fn main() {}
