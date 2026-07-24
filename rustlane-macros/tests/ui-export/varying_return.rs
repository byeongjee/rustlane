#![feature(portable_simd)]
#![allow(unused)]
use rustlane::export;
use rustlane::prelude::*;

#[export]
fn bad(n: i32) -> Varying<i32> {
    Varying::splat(n)
}

fn main() {}
