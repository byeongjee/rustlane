#![feature(portable_simd)]
#![allow(unused)]
use spmd::prelude::*;
use spmd::{kernel, SpmdValue};

#[derive(SpmdValue, Clone, Copy)]
#[repr(C)]
struct Hit {
    t: f32,
    #[spmd(uniform)]
    id: i32,
}

#[kernel]
fn bad(sel: Varying<f32>, h: VaryingHit<N>, g: VaryingHit<N>) -> Varying<f32> {
    let mut p = h;
    if sel > 0.0 {
        p = g;
    }
    p.t
}

fn main() {}
