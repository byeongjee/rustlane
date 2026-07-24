//! Proc macros for the `rustlane` SPMD-on-SIMD library.
//!
//! `#[kernel]` rewrites natural scalar-looking Rust (comparisons, `if`,
//! `while`, `break`, indexing, ...) into the mask-threaded trait calls of
//! `rustlane-core`.
//!
//! The kernel-world macros (`foreach!`, `foreach_2d!`, `foreach_tiled!`,
//! `unmasked!`, `cif!`, `cwhile!`) are expanded INLINE by `#[kernel]` itself;
//! the function-like macros exported here are placeholders that produce a
//! helpful error when invoked outside a kernel.

mod export;
mod kernel;
mod rewrite;
mod scan;
mod spmd_value;

use proc_macro::TokenStream;

/// Turns a scalar-looking fn (or inherent impl block) into a width- and
/// exec-generic SPMD kernel. The ABI:
///
/// ```ignore
/// #[kernel]
/// fn f(x: Varying<f32>, n: i32) -> Varying<f32> { .. }
/// // becomes
/// fn f<const N: usize, __E>(__exec: __E, x: Varying<f32, N>, n: i32) -> Varying<f32, N>
/// where __E: rustlane::Exec + rustlane::ActiveMask<N>, LaneCount<N>: SupportedLaneCount { .. }
/// ```
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    kernel::expand(attr.into(), item.into()).into()
}

/// Turns an all-uniform-signature fn into a runtime-dispatched entry point:
/// the body is the
/// `#[kernel]` transform, instantiated behind one `#[target_feature]` shim per
/// target (x86-64: sse2/sse4.1/avx2+fma/avx512f; aarch64: neon) with an
/// `OnceLock` + `is_x86_feature_detected!` dispatcher. `Varying` in the
/// signature is a compile error (its width cannot cross the dispatch boundary).
/// Override the target set with `#[rustlane::export(targets("avx2", "avx512"))]`.
#[proc_macro_attribute]
pub fn export(attr: TokenStream, item: TokenStream) -> TokenStream {
    export::expand(attr.into(), item.into()).into()
}

/// Derives the SoA varying representation for a `#[repr(C)]` value struct.
/// `#[derive(SpmdValue)] struct S { .. }` generates `VaryingS<N>`
/// (field-wise varying, recursing through nested `SpmdValue` fields), the
/// `SpmdValue`/`SpmdGather` trait impls, field-wise `MaskedAssign` across the
/// exec contexts, and inherent `select`/`gather`. `#[spmd(uniform)]` keeps a
/// field scalar inside `VaryingS` (its whole-struct varying `MaskedAssign` /
/// `select` are then not generated — a missing-impl error under varying masks).
#[proc_macro_derive(SpmdValue, attributes(spmd))]
pub fn derive_spmd_value(item: TokenStream) -> TokenStream {
    spmd_value::expand(item.into()).into()
}

macro_rules! kernel_only_stub {
    ($($(#[$doc:meta])* $name:ident;)*) => {$(
        $(#[$doc])*
        #[proc_macro]
        pub fn $name(input: TokenStream) -> TokenStream {
            let _ = input;
            let msg = concat!(
                "`", stringify!($name),
                "!` is only supported inside a #[rustlane::kernel] function"
            );
            quote::quote!(compile_error!(#msg);).into()
        }
    )*};
}

kernel_only_stub! {
    /// `foreach!(i in 0..n { .. })` — parallel iteration over a `usize`
    /// range; `i` is a `LinearIndex<N>` (contiguous loads/stores). Main
    /// chunks run mask-free, the non-divisible tail runs masked.
    foreach;
    /// `foreach_2d!(y in 0..h, x in 0..w { .. })` — row-major nesting of
    /// `foreach!`; `y` is a uniform `usize`, `x` a `LinearIndex<N>`.
    foreach_2d;
    /// `foreach_tiled!(y in 0..h, x in 0..w { .. })` — ISPC-order tiled
    /// iteration; `y`/`x` are `Varying<i32, N>` coordinates.
    foreach_tiled;
    /// `unmasked! { .. }` — statements execute with ALL lanes on.
    /// `break`/`continue`/`return` inside are errors.
    unmasked;
    /// `cif!(cond => { .. } else { .. })` — coherent `if`: same lowering as
    /// `if` but the branch guard is `any()` (skips empty-mask bodies).
    cif;
    /// `cwhile!(cond => { .. })` — coherent `while`; identical lowering to
    /// `while` (its exit check is already the single `any()`).
    cwhile;
}
