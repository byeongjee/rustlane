use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse2, punctuated::Punctuated, spanned::Spanned, Attribute, Data, DeriveInput, Error, Fields,
    Ident, Meta, Token, Type, Visibility,
};

enum FieldKind {
    Varying,
    Uniform,
}

struct FieldInfo {
    ident: Ident,
    ty: Type,
    kind: FieldKind,
}

pub fn expand(input: TokenStream) -> TokenStream {
    let di: DeriveInput = match parse2(input) {
        Ok(d) => d,
        Err(e) => return e.to_compile_error(),
    };

    let mut errors: Vec<Error> = Vec::new();

    if !di.generics.params.is_empty() {
        errors.push(Error::new(
            di.generics.span(),
            "#[derive(SpmdValue)] does not support generic value structs \
             (the varying representation owns `const N: usize`)",
        ));
    }

    if !has_repr_c(&di.attrs) {
        errors.push(Error::new(
            di.ident.span(),
            "#[derive(SpmdValue)] requires `#[repr(C)]` on the value struct \
             (its field offsets back the AoS gather)",
        ));
    }

    let fields = match &di.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(n) => &n.named,
            _ => {
                errors.push(Error::new(
                    di.ident.span(),
                    "#[derive(SpmdValue)] requires a struct with named fields",
                ));
                return combine(errors);
            }
        },
        _ => {
            errors.push(Error::new(
                di.ident.span(),
                "#[derive(SpmdValue)] applies to structs only",
            ));
            return combine(errors);
        }
    };

    let mut infos: Vec<FieldInfo> = Vec::new();
    for f in fields {
        let ident = f.ident.clone().expect("named field");
        let kind = match uniform_attr(&f.attrs) {
            Ok(true) => FieldKind::Uniform,
            Ok(false) => FieldKind::Varying,
            Err(e) => {
                errors.push(e);
                FieldKind::Varying
            }
        };
        infos.push(FieldInfo {
            ident,
            ty: f.ty.clone(),
            kind,
        });
    }

    if !errors.is_empty() {
        return combine(errors);
    }

    emit(&di.ident, &di.vis, &infos)
}

fn combine(errors: Vec<Error>) -> TokenStream {
    errors.into_iter().map(|e| e.to_compile_error()).collect()
}

fn has_repr_c(attrs: &[Attribute]) -> bool {
    for a in attrs {
        if !a.path().is_ident("repr") {
            continue;
        }
        if let Ok(metas) = a.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
            if metas.iter().any(|m| m.path().is_ident("C")) {
                return true;
            }
        }
    }
    false
}

fn uniform_attr(attrs: &[Attribute]) -> Result<bool, Error> {
    let mut uniform = false;
    for a in attrs {
        if !a.path().is_ident("spmd") {
            continue;
        }
        a.parse_nested_meta(|m| {
            if m.path.is_ident("uniform") {
                uniform = true;
                Ok(())
            } else {
                Err(m.error("unknown `#[spmd(..)]` field option (expected `uniform`)"))
            }
        })?;
    }
    Ok(uniform)
}

fn emit(name: &Ident, vis: &Visibility, fields: &[FieldInfo]) -> TokenStream {
    let vname = format_ident!("Varying{}", name);
    let all_varying = fields.iter().all(|f| matches!(f.kind, FieldKind::Varying));

    let struct_fields = fields.iter().map(|f| {
        let id = &f.ident;
        let ty = &f.ty;
        match f.kind {
            FieldKind::Varying => quote!(pub #id: <#ty as ::rustlane::SpmdValue>::Varying<N>),
            FieldKind::Uniform => quote!(pub #id: #ty),
        }
    });

    let splat_fields = fields.iter().map(|f| {
        let id = &f.ident;
        match f.kind {
            FieldKind::Varying => quote!(#id: ::rustlane::SpmdValue::splat(self.#id)),
            FieldKind::Uniform => quote!(#id: self.#id),
        }
    });

    let varying_struct = quote! {
        #[derive(::core::clone::Clone, ::core::marker::Copy)]
        #[allow(non_camel_case_types)]
        #vis struct #vname<const N: usize> {
            #(#struct_fields,)*
        }
    };

    let impl_spmd_value = quote! {
        impl ::rustlane::SpmdValue for #name {
            type Varying<const N: usize> = #vname<N>;

            #[inline(always)]
            fn splat<const N: usize>(self) -> #vname<N> {
                #vname { #(#splat_fields,)* }
            }
        }
    };

    let assign_stmts: Vec<TokenStream> = fields
        .iter()
        .map(|f| {
            let id = &f.ident;
            match f.kind {
                FieldKind::Varying => {
                    quote!(::rustlane::MaskedAssign::masked_assign(&mut self.#id, __exec, value.#id);)
                }
                FieldKind::Uniform => quote!(self.#id = value.#id;),
            }
        })
        .collect();

    let masked_assign_impl = |ctx: TokenStream| {
        let stmts = &assign_stmts;
        quote! {
            impl<const N: usize> ::rustlane::MaskedAssign<#ctx> for #vname<N> {
                #[inline(always)]
                fn masked_assign(&mut self, __exec: #ctx, value: Self) {
                    #(#stmts)*
                }
            }
        }
    };
    let mut masked_assign = TokenStream::new();
    masked_assign.extend(masked_assign_impl(quote!(::rustlane::AllOn)));
    masked_assign.extend(masked_assign_impl(quote!(::rustlane::BoolGuard)));
    if all_varying {
        masked_assign.extend(masked_assign_impl(quote!(::rustlane::VMask<N>)));
        masked_assign.extend(masked_assign_impl(quote!(::rustlane::VMaskGuard<N>)));
    }

    let gather_and_select = if all_varying {
        let gather_fields = fields.iter().map(|f| {
            let id = &f.ident;
            let ty = &f.ty;
            quote! {
                #id: unsafe {
                    <#ty as ::rustlane::SpmdGather>::gather_fields::<__Base, N, __E>(
                        __base,
                        __idx,
                        __field_offset + ::core::mem::offset_of!(#name, #id),
                        __exec,
                    )
                }
            }
        });
        let select_fields = fields.iter().map(|f| {
            let id = &f.ident;
            quote!(#id: self.#id.select(__mask, __other.#id))
        });
        quote! {
            impl ::rustlane::SpmdGather for #name {
                #[inline(always)]
                unsafe fn gather_fields<__Base, const N: usize, __E>(
                    __base: &[__Base],
                    __idx: ::rustlane::Varying<i32, N>,
                    __field_offset: usize,
                    __exec: __E,
                ) -> #vname<N>
                where
                    __E: ::rustlane::ActiveMask<N> + ::core::marker::Copy,
                {
                    #vname { #(#gather_fields,)* }
                }
            }

            impl<const N: usize> #vname<N> {
                /// AoS gather: read one `#name` per lane out of `base` at the
                /// `Varying<i32, N>` element indices (one strided gather per
                /// leaf field; inactive/out-of-bounds lanes never addressed).
                #[inline(always)]
                pub fn gather<__E: ::rustlane::ActiveMask<N> + ::core::marker::Copy>(
                    base: &[#name],
                    idx: ::rustlane::Varying<i32, N>,
                    exec: __E,
                ) -> Self {
                    unsafe {
                        <#name as ::rustlane::SpmdGather>::gather_fields::<#name, N, __E>(base, idx, 0, exec)
                    }
                }

                /// Field-wise lane select: `self` where `mask` is set, `other`
                /// elsewhere (`Mask<i32, N>` condition currency).
                #[inline(always)]
                pub fn select(self, __mask: ::core::simd::Mask<i32, N>, __other: Self) -> Self {
                    #vname { #(#select_fields,)* }
                }
            }
        }
    } else {
        TokenStream::new()
    };

    quote! {
        #varying_struct
        #impl_spmd_value
        #masked_assign
        #gather_and_select
    }
}
