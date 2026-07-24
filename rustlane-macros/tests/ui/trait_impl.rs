#![feature(portable_simd)]
#![allow(unused)]
use rustlane::kernel;
use rustlane::prelude::*;

struct Ops;
trait Doubler {
    fn go(x: Varying<i32, 8>) -> Varying<i32, 8>;
}

#[kernel]
impl Doubler for Ops {
    fn go(x: Varying<i32>) -> Varying<i32> {
        x * 2
    }
}

fn main() {}
