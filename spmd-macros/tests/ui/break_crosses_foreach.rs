#![feature(portable_simd)]
#![allow(unused)]
use spmd::kernel;
use spmd::prelude::*;

#[kernel]
fn bad(a: &[i32], out: &mut [i32]) {
    loop {
        foreach!(i in 0..a.len() {
            out[i] = a[i];
            break;
        })
    }
}

fn main() {}
