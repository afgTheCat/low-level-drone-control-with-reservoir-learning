use proc_macro::TokenStream;
use quote::quote;
use syn::{Ident, ItemStruct, parse_macro_input};

#[proc_macro_attribute]
pub fn data_vars(attr: TokenStream, input: TokenStream) -> TokenStream {
    let const_ident = parse_macro_input!(attr as Ident);
    let item_struct = parse_macro_input!(input as ItemStruct);
    let field_count = item_struct.fields.iter().count();

    let name = &item_struct.ident;

    let output = quote! {
        #item_struct

        impl #name {
            pub const #const_ident: usize = #field_count;
        }
    };

    TokenStream::from(output)
}
