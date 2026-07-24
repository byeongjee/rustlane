
mod export;
mod kernel;
mod rewrite;
mod scan;
mod spmd_value;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    kernel::expand(attr.into(), item.into()).into()
}

#[proc_macro_attribute]
pub fn export(attr: TokenStream, item: TokenStream) -> TokenStream {
    export::expand(attr.into(), item.into()).into()
}

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
                "!` is only supported inside a #[spmd::kernel] function"
            );
            quote::quote!(compile_error!(#msg);).into()
        }
    )*};
}

kernel_only_stub! {
    foreach;
    foreach_2d;
    foreach_tiled;
    unmasked;
    cif;
    cwhile;
}
