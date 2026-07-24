
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::visit_mut::{self, VisitMut};
use syn::{parse2, Error, FnArg, Ident, ItemFn, Pat, PathSegment, ReturnType, Type};

use crate::kernel::{add_kernel_attrs, combine_errors, kernel_body, ret_mode_of, transform_sig};
use crate::rewrite;


#[derive(Clone, Copy, PartialEq)]
enum Arch {
    X86_64,
    Aarch64,
}

#[derive(Clone, Copy)]
struct Target {
    key: &'static str,
    arch: Arch,
    features: &'static [&'static str],
    detect: &'static [&'static str],
    n: usize,
}

const X86_TARGETS: &[Target] = &[
    Target { key: "avx512f", arch: Arch::X86_64, features: &["avx512f"], detect: &["avx512f"], n: 16 },
    Target { key: "avx2", arch: Arch::X86_64, features: &["avx2", "fma"], detect: &["avx2", "fma"], n: 8 },
    Target { key: "sse41", arch: Arch::X86_64, features: &["sse4.1"], detect: &["sse4.1"], n: 4 },
    Target { key: "sse2", arch: Arch::X86_64, features: &[], detect: &[], n: 4 },
];

const NEON: Target =
    Target { key: "neon", arch: Arch::Aarch64, features: &[], detect: &[], n: 8 };

fn lookup_target(name: &str) -> Option<Target> {
    match name {
        "sse2" => Some(X86_TARGETS[3]),
        "sse4.1" | "sse41" | "sse4_1" => Some(X86_TARGETS[2]),
        "avx2" => Some(X86_TARGETS[1]),
        "avx512" | "avx512f" => Some(X86_TARGETS[0]),
        "neon" => Some(NEON),
        _ => None,
    }
}


fn parse_targets(attr: TokenStream) -> Result<Vec<Target>, Error> {
    if attr.is_empty() {
        let mut v: Vec<Target> = X86_TARGETS.to_vec();
        v.push(NEON);
        return Ok(v);
    }
    let meta: syn::MetaList = parse2(attr)?;
    if !meta.path.is_ident("targets") {
        return Err(Error::new_spanned(
            &meta.path,
            "#[rustlane::export] only accepts `targets(\"..\", ..)`",
        ));
    }
    let names = meta.parse_args_with(
        syn::punctuated::Punctuated::<syn::LitStr, syn::Token![,]>::parse_terminated,
    )?;
    if names.is_empty() {
        return Err(Error::new_spanned(
            &meta,
            "`targets(..)` needs at least one target name",
        ));
    }
    let mut chosen_x86: Vec<Target> = Vec::new();
    let mut keep_neon = false;
    for lit in &names {
        let name = lit.value();
        match lookup_target(&name) {
            Some(t) if t.arch == Arch::Aarch64 => keep_neon = true,
            Some(t) => {
                if !chosen_x86.iter().any(|c| c.key == t.key) {
                    chosen_x86.push(t);
                }
            }
            None => {
                return Err(Error::new_spanned(
                    lit,
                    "unknown target; known: sse2, sse4.1, avx2, avx512, neon",
                ))
            }
        }
    }
    let mut out: Vec<Target> = X86_TARGETS
        .iter()
        .copied()
        .filter(|t| chosen_x86.iter().any(|c| c.key == t.key))
        .collect();
    let _ = keep_neon;
    out.push(NEON);
    Ok(out)
}


struct VaryingFinder {
    hit: Option<Span>,
}

impl VisitMut for VaryingFinder {
    fn visit_path_segment_mut(&mut self, seg: &mut PathSegment) {
        if self.hit.is_none() && seg.ident == "Varying" {
            self.hit = Some(seg.ident.span());
        }
        visit_mut::visit_path_segment_mut(self, seg);
    }
}

fn check_no_varying(ty: &Type, errors: &mut Vec<Error>) {
    let mut f = VaryingFinder { hit: None };
    f.visit_type_mut(&mut ty.clone());
    if let Some(sp) = f.hit {
        errors.push(Error::new(
            sp,
            "`Varying` cannot appear in an #[rustlane::export] signature: its in-memory \
             width is chosen by the runtime target dispatcher, so it cannot cross the \
             export boundary. Take scalars / `&[T]` / `&mut [T]` and iterate with \
             `foreach!` inside the body (each lane's varying data stays within one target's \
             width).",
        ));
    }
}

fn check_no_generics(func: &ItemFn, errors: &mut Vec<Error>) {
    for gp in &func.sig.generics.params {
        errors.push(Error::new_spanned(
            gp,
            "#[rustlane::export] fns take no generic parameters (use elided lifetimes on \
             `&[T]`/`&mut [T]` args)",
        ));
    }
}

fn collect_args(func: &ItemFn) -> (Vec<Ident>, Vec<Type>) {
    let mut names = Vec::new();
    let mut types = Vec::new();
    for arg in &func.sig.inputs {
        if let FnArg::Typed(pt) = arg {
            if let Pat::Ident(pi) = &*pt.pat {
                names.push(pi.ident.clone());
                types.push((*pt.ty).clone());
            }
        }
    }
    (names, types)
}


pub fn expand(attr: TokenStream, item: TokenStream) -> TokenStream {
    let targets = match parse_targets(attr) {
        Ok(t) => t,
        Err(e) => return e.to_compile_error(),
    };
    let func = match parse2::<ItemFn>(item) {
        Ok(f) => f,
        Err(_) => {
            return Error::new(
                Span::call_site(),
                "#[rustlane::export] applies to free functions with an all-uniform signature",
            )
            .to_compile_error();
        }
    };

    let mut errors = Vec::new();
    check_no_generics(&func, &mut errors);
    for arg in &func.sig.inputs {
        if let FnArg::Typed(pt) = arg {
            check_no_varying(&pt.ty, &mut errors);
        }
    }
    if let ReturnType::Type(_, ty) = &func.sig.output {
        check_no_varying(ty, &mut errors);
    }

    let mut impl_sig = func.sig.clone();
    transform_sig(&mut impl_sig, &mut errors, false);
    if !errors.is_empty() {
        return combine_errors(errors);
    }

    let name = func.sig.ident.clone();
    let vis = func.vis.clone();
    let user_attrs = func.attrs.clone();
    let inputs = func.sig.inputs.clone();
    let output = func.sig.output.clone();
    let (arg_names, arg_types) = collect_args(&func);

    let impl_name = format_ident!("__{}_impl", name);
    impl_sig.ident = impl_name.clone();
    let ret_mode = ret_mode_of(&func.sig.output);
    let (stmts, rw_errors) = rewrite::rewrite_body(*func.block, ret_mode);
    if !rw_errors.is_empty() {
        return combine_errors(rw_errors);
    }

    let mut impl_attrs = Vec::new();
    add_kernel_attrs(&mut impl_attrs);
    let impl_body = kernel_body(&stmts);

    let dispatch = gen_dispatch(
        &name, &impl_name, &vis, &user_attrs, &inputs, &output, &arg_names, &arg_types, &targets,
    );

    quote! {
        #(#impl_attrs)*
        #impl_sig #impl_body

        #dispatch
    }
}


#[allow(clippy::too_many_arguments)]
fn gen_dispatch(
    name: &Ident,
    impl_name: &Ident,
    vis: &syn::Visibility,
    user_attrs: &[syn::Attribute],
    inputs: &syn::punctuated::Punctuated<FnArg, syn::Token![,]>,
    output: &ReturnType,
    arg_names: &[Ident],
    arg_types: &[Type],
    targets: &[Target],
) -> TokenStream {
    let x86: Vec<&Target> = targets.iter().filter(|t| t.arch == Arch::X86_64).collect();
    let arm: Vec<&Target> = targets.iter().filter(|t| t.arch == Arch::Aarch64).collect();

    let mut x86_shims: Vec<TokenStream> = Vec::new();
    for t in &x86 {
        let shim = format_ident!("__{}_{}", name, t.key);
        let n = t.n;
        let call = quote! { #impl_name::<#n, _>(::rustlane::AllOn, #(#arg_names),*) };
        if t.features.is_empty() {
            x86_shims.push(quote! {
                #[cfg(target_arch = "x86_64")]
                #[inline(never)]
                fn #shim(#inputs) #output { #call }
            });
        } else {
            let feat = t.features.join(",");
            x86_shims.push(quote! {
                #[cfg(target_arch = "x86_64")]
                #[inline(never)]
                #[target_feature(enable = #feat)]
                unsafe fn #shim(#inputs) #output { #call }
            });
        }
    }

    let fn_ty = format_ident!("__{}Fn", name);
    let resolve = format_ident!("__{}_resolve", name);

    let (x86_support, x86_wrapper_body): (TokenStream, TokenStream) = if x86.is_empty() {
        (
            quote! {},
            quote! {
                compile_error!(
                    "#[rustlane::export]: no x86_64 target in `targets(..)`; add one (e.g. \"avx2\")"
                );
            },
        )
    } else {
        let mut probes: Vec<TokenStream> = Vec::new();
        for t in &x86[..x86.len() - 1] {
            let shim = format_ident!("__{}_{}", name, t.key);
            let checks = t.detect.iter().map(|d| quote! { is_x86_feature_detected!(#d) });
            probes.push(quote! {
                if #(#checks)&&* { return #shim; }
            });
        }
        let last = format_ident!("__{}_{}", name, x86[x86.len() - 1].key);

        let support = quote! {
            #[cfg(target_arch = "x86_64")]
            #[allow(non_camel_case_types)]
            type #fn_ty = unsafe fn(#(#arg_types),*) #output;

            #[cfg(target_arch = "x86_64")]
            fn #resolve() -> #fn_ty {
                #(#probes)*
                #last
            }
        };
        let wrapper = quote! {
            static __DISPATCH: ::std::sync::OnceLock<#fn_ty> = ::std::sync::OnceLock::new();
            let __f = *__DISPATCH.get_or_init(#resolve);
            return unsafe { __f(#(#arg_names),*) };
        };
        (support, wrapper)
    };

    let arm_wrapper_body: TokenStream = match arm.first() {
        Some(t) => {
            let n = t.n;
            quote! { return #impl_name::<#n, _>(::rustlane::AllOn, #(#arg_names),*); }
        }
        None => quote! {
            compile_error!("#[rustlane::export]: no aarch64 target selected");
        },
    };

    quote! {
        #(#x86_shims)*

        #x86_support

        #(#user_attrs)*
        #vis fn #name(#inputs) #output {
            #[cfg(target_arch = "x86_64")]
            {
                #x86_wrapper_body
            }
            #[cfg(target_arch = "aarch64")]
            {
                #arm_wrapper_body
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                unimplemented!("#[rustlane::export] supports x86_64 and aarch64 targets")
            }
        }
    }
}
