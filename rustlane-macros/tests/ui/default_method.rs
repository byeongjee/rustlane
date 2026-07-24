#![feature(portable_simd)]
#![allow(unused)]
use rustlane::kernel;
use rustlane::prelude::*;

struct Ops;

#[kernel]
impl Ops {
    default fn go(x: Varying<i32>) -> Varying<i32> {
        x
    }
}

fn main() {}
