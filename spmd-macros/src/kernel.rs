
use proc_macro2::TokenStream;
use quote::{quote, ToTokens};
use syn::visit_mut::VisitMut;
use syn::{
    parse_quote, Attribute, Error, FnArg, GenericParam, Ident, ImplItem, Item, ItemFn,
    ItemImpl, Pat, PathArguments, PathSegment, ReturnType, Signature, Stmt, Type,
};

use crate::rewrite::{self, RetMode};


pub fn expand(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return Error::new_spanned(&attr, "#[spmd::kernel] takes no arguments")
            .to_compile_error();
    }
    match syn::parse2::<Item>(item) {
        Ok(Item::Fn(f)) => expand_fn(f),
        Ok(Item::Impl(i)) => expand_impl(i),
        Ok(other) => Error::new_spanned(
            &other,
            "#[spmd::kernel] applies to free functions and inherent `impl` blocks",
        )
        .to_compile_error(),
        Err(e) => e.to_compile_error(),
    }
}

fn expand_fn(f: ItemFn) -> TokenStream {
    let ItemFn {
        mut attrs,
        vis,
        mut sig,
        block,
    } = f;
    let mut errors = Vec::new();
    transform_sig(&mut sig, &mut errors, false);
    let ret_mode = ret_mode_of(&sig.output);
    let (stmts, rw_errors) = rewrite::rewrite_body(*block, ret_mode);
    errors.extend(rw_errors);
    if !errors.is_empty() {
        return combine_errors(errors);
    }
    add_kernel_attrs(&mut attrs);
    let body = kernel_body(&stmts);
    quote! {
        #(#attrs)*
        #vis #sig #body
    }
}

fn expand_impl(imp: ItemImpl) -> TokenStream {
    let mut errors = Vec::new();
    if let Some((_, path, _)) = &imp.trait_ {
        return Error::new_spanned(
            path,
            "#[spmd::kernel] impl blocks must be inherent impls (trait impls would change the \
             trait's method signatures)",
        )
        .to_compile_error();
    }
    let ItemImpl {
        attrs,
        generics,
        self_ty,
        items,
        ..
    } = imp;
    let mut out_items: Vec<TokenStream> = Vec::new();
    for item in items {
        match item {
            ImplItem::Fn(m) => {
                let syn::ImplItemFn {
                    mut attrs,
                    vis,
                    defaultness,
                    mut sig,
                    block,
                } = m;
                if defaultness.is_some() {
                    errors.push(Error::new_spanned(
                        &defaultness,
                        "`default` methods are not supported in #[spmd::kernel] impls",
                    ));
                }
                transform_sig(&mut sig, &mut errors, true);
                let ret_mode = ret_mode_of(&sig.output);
                let (stmts, rw_errors) = rewrite::rewrite_body(block, ret_mode);
                errors.extend(rw_errors);
                add_kernel_attrs(&mut attrs);
                let body = kernel_body(&stmts);
                out_items.push(quote! {
                    #(#attrs)*
                    #vis #sig #body
                });
            }
            other => out_items.push(other.to_token_stream()),
        }
    }
    if !errors.is_empty() {
        return combine_errors(errors);
    }
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    quote! {
        #(#attrs)*
        impl #impl_generics #self_ty #where_clause {
            #(#out_items)*
        }
    }
}


pub(crate) fn kernel_body(stmts: &[Stmt]) -> TokenStream {
    quote! {
        {
            #[allow(unused_imports)]
            use ::spmd::prelude::*;
            if <__E as ::spmd::Exec>::UNIFORM {
                let __exec = ::spmd::AllOn;
                #(#stmts)*
            } else {
                let __exec = ::spmd::VMask::<N>(::spmd::ActiveMask::<N>::active_mask(__exec));
                #(#stmts)*
            }
        }
    }
}

pub(crate) fn add_kernel_attrs(attrs: &mut Vec<Attribute>) {
    let has_inline = attrs.iter().any(|a| a.path().is_ident("inline"));
    if !has_inline {
        attrs.push(parse_quote!(#[inline(always)]));
    }
    attrs.push(parse_quote!(#[allow(unused_parens, non_camel_case_types, clippy::all)]));
}

pub(crate) fn transform_sig(sig: &mut Signature, errors: &mut Vec<Error>, allow_receiver: bool) {
    if let Some(a) = &sig.asyncness {
        errors.push(Error::new_spanned(a, "#[kernel] fns cannot be `async`"));
    }
    if let Some(u) = &sig.unsafety {
        errors.push(Error::new_spanned(u, "#[kernel] fns cannot be `unsafe`"));
    }
    if let Some(abi) = &sig.abi {
        errors.push(Error::new_spanned(abi, "#[kernel] fns cannot have an ABI"));
    }
    if let Some(v) = &sig.variadic {
        errors.push(Error::new_spanned(v, "#[kernel] fns cannot be variadic"));
    }
    for gp in &sig.generics.params {
        match gp {
            GenericParam::Lifetime(_) => {}
            other => errors.push(Error::new_spanned(
                other,
                "#[kernel] fns cannot declare their own type/const generics in v1 (the macro \
                 owns `const N: usize` and the exec parameter)",
            )),
        }
    }

    sig.generics.params.push(parse_quote!(const N: usize));
    sig.generics.params.push(parse_quote!(__E));
    let wc = sig.generics.make_where_clause();
    wc.predicates
        .push(parse_quote!(__E: ::spmd::Exec + ::spmd::ActiveMask<N>));
    wc.predicates
        .push(parse_quote!(::core::simd::LaneCount<N>: ::core::simd::SupportedLaneCount));

    let mut has_receiver = false;
    for arg in sig.inputs.iter_mut() {
        match arg {
            FnArg::Receiver(r) => {
                if !allow_receiver {
                    errors.push(Error::new_spanned(
                        &*r,
                        "`self` is only allowed in #[kernel] impl blocks",
                    ));
                }
                has_receiver = true;
            }
            FnArg::Typed(pt) => {
                match &*pt.pat {
                    Pat::Ident(pi) => {
                        if let Some(msg) = check_reserved(&pi.ident) {
                            errors.push(Error::new(pi.ident.span(), msg));
                        }
                    }
                    other => errors.push(Error::new_spanned(
                        other,
                        "#[kernel] parameters must be plain identifiers",
                    )),
                }
                widen_type(&mut pt.ty);
            }
        }
    }
    let pos = usize::from(has_receiver);
    sig.inputs.insert(pos, parse_quote!(__exec: __E));

    if let ReturnType::Type(_, ty) = &mut sig.output {
        widen_type(ty);
    }
}

pub(crate) fn ret_mode_of(output: &ReturnType) -> RetMode {
    match output {
        ReturnType::Default => RetMode::Unit,
        ReturnType::Type(_, ty) => match &**ty {
            Type::Tuple(t) if t.elems.is_empty() => RetMode::Unit,
            _ => RetMode::Value(ty.clone()),
        },
    }
}

pub(crate) fn combine_errors(errors: Vec<Error>) -> TokenStream {
    errors.into_iter().map(|e| e.to_compile_error()).collect()
}


struct VaryingWiden;

impl VisitMut for VaryingWiden {
    fn visit_path_segment_mut(&mut self, seg: &mut PathSegment) {
        syn::visit_mut::visit_path_segment_mut(self, seg);
        if seg.ident == "Varying" {
            if let PathArguments::AngleBracketed(ab) = &mut seg.arguments {
                let single_type =
                    ab.args.len() == 1 && matches!(&ab.args[0], syn::GenericArgument::Type(_));
                if single_type {
                    ab.args.push(parse_quote!(N));
                }
            }
        }
    }
}

pub fn widen_type(t: &mut Type) {
    VaryingWiden.visit_type_mut(t);
}

pub fn widen_path(p: &mut syn::Path) {
    VaryingWiden.visit_path_mut(p);
}

pub fn check_reserved(id: &Ident) -> Option<&'static str> {
    let s = id.to_string();
    if s == "N" {
        Some("`N` is reserved inside #[kernel]: it names the lane-count const parameter \
              (you may read it, but not bind it)")
    } else if s.starts_with("__") {
        Some("identifiers starting with `__` are reserved by #[kernel] machinery")
    } else {
        None
    }
}
