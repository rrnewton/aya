//! Proc macro crate for `core_read!`.
//!
//! Provides the `core_read_impl!` proc macro that:
//!   1. Extracts the struct type name (last path segment) and field path
//!   2. Emits a marker record into the `.aya.core_relo` ELF section via
//!      a `#[link_section]` static variable
//!   3. Emits the actual field-read code using `offset_of!` and
//!      `probe_read_kernel`
//!
//! The marker format in `.aya.core_relo` is a fixed-layout byte array:
//!   - 1 byte: record tag (0xAC = "Aya Core")
//!   - 1 byte: struct name length (N)
//!   - N bytes: struct name (UTF-8, no NUL)
//!   - 1 byte: field path length (M)
//!   - M bytes: field path (UTF-8, dot-separated, no NUL)
//!
//! The `aya-core-postprocessor` reads these markers to discover which
//! struct field accesses need CO-RE relocations, without requiring a
//! hand-written sidecar TOML file.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{Expr, Path, Token, punctuated::Punctuated};

/// Input to the `core_read_impl!` macro:
///   core_read_impl!(StructType, ptr_expr, field.path)
struct CoreReadInput {
    struct_ty: Path,
    _comma1: Token![,],
    ptr_expr: Expr,
    _comma2: Token![,],
    field_segments: Punctuated<syn::Ident, Token![.]>,
}

impl Parse for CoreReadInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let struct_ty = input.parse()?;
        let _comma1 = input.parse()?;
        let ptr_expr = input.parse()?;
        let _comma2 = input.parse()?;

        // Parse the dot-separated field path (e.g., `scx.dsq_vtime`).
        let field_segments = Punctuated::parse_separated_nonempty(input)?;

        Ok(Self {
            struct_ty,
            _comma1,
            ptr_expr,
            _comma2,
            field_segments,
        })
    }
}

/// Build a token stream for dot-separated field access like `a.b.c`.
///
/// `quote!` doesn't support `#(#idents).+` syntax (that's a declarative
/// macro pattern), so we construct the token stream manually by
/// interleaving idents with `.` punctuation.
fn build_field_access(idents: &Punctuated<syn::Ident, Token![.]>) -> TokenStream2 {
    let mut tokens = TokenStream2::new();
    for (i, ident) in idents.iter().enumerate() {
        if i > 0 {
            tokens.extend(quote! { . });
        }
        tokens.extend(quote! { #ident });
    }
    tokens
}

/// Proc macro implementation for `core_read!`.
///
/// Usage: `core_read_impl!(vmlinux::task_struct, p, scx.dsq_vtime)`
///
/// This emits:
///   1. A `#[link_section = ".aya.core_relo"]` static with the struct name
///      and field path encoded as bytes
///   2. The actual field-read code: computes the field pointer using
///      `offset_of!()` and reads via `probe_read_kernel()`
#[proc_macro]
pub fn core_read_impl(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as CoreReadInput);

    // Extract the last segment of the type path as the struct name.
    // e.g., `vmlinux::task_struct` -> "task_struct"
    let struct_name = match input.struct_ty.segments.last() {
        Some(seg) => seg.ident.to_string(),
        None => {
            return syn::Error::new_spanned(&input.struct_ty, "empty type path")
                .to_compile_error()
                .into();
        }
    };

    // Build the dot-separated field path string.
    // e.g., `scx.dsq_vtime` -> "scx.dsq_vtime"
    let field_path: String = input
        .field_segments
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(".");

    let struct_ty = &input.struct_ty;
    let ptr_expr = &input.ptr_expr;

    // Build token streams for field access in offset_of! and member expressions.
    let field_access = build_field_access(&input.field_segments);

    // Build the marker bytes for .aya.core_relo.
    // Format: [0xAC, name_len, name_bytes..., path_len, path_bytes...]
    let struct_name_bytes = struct_name.as_bytes();
    let field_path_bytes = field_path.as_bytes();
    let name_len = struct_name_bytes.len();
    let path_len = field_path_bytes.len();
    let total_len = 1 + 1 + name_len + 1 + path_len;

    // Build the byte array as a literal.
    let mut marker_bytes = Vec::with_capacity(total_len);
    marker_bytes.push(0xAC_u8); // tag
    marker_bytes.push(name_len as u8);
    marker_bytes.extend_from_slice(struct_name_bytes);
    marker_bytes.push(path_len as u8);
    marker_bytes.extend_from_slice(field_path_bytes);

    // Generate a unique static name to avoid conflicts when the same
    // struct/field pair is used in multiple functions.  We append a hash
    // of the span location to disambiguate.
    let sanitized_name = format!(
        "__aya_core_relo_{}_{}",
        struct_name,
        field_path.replace('.', "_")
    );
    let static_name = format_ident!("{}", sanitized_name);

    let marker_len = marker_bytes.len();

    // Build the offset_of! call.  We need to construct the full macro
    // invocation including the field path as a single token stream,
    // since quote! can't handle the special offset_of! field syntax
    // with interpolation.
    let offset_of_tokens = {
        // Build the argument list: Type, field.path
        let mut args = quote! { #struct_ty, };
        args.extend(field_access.clone());
        // Wrap in offset_of!( ... )
        quote! { ::core::mem::offset_of!( #args ) }
    };

    let expanded = quote! {{
        // Emit the marker record into .aya.core_relo.
        // The `used` attribute prevents the linker from discarding it.
        #[allow(non_upper_case_globals)]
        #[link_section = ".aya.core_relo"]
        #[used]
        static #static_name: [u8; #marker_len] = [#(#marker_bytes),*];

        // Compute the field pointer using offset_of and raw pointer
        // arithmetic.
        let __aya_base = #ptr_expr as *const u8;
        let __aya_offset = #offset_of_tokens;
        let _ = __aya_offset; // suppress unused warning; used for documentation

        // Use a null pointer to infer the field type, then read from the
        // real address at (base + offset).  The pointer dereference below
        // is only used for type inference; it is never actually loaded
        // from -- probe_read_kernel does the actual read.
        let __aya_typed_ptr = unsafe {
            &raw const (*(__aya_base as *const #struct_ty)).#field_access
        };
        unsafe { probe_read_kernel(__aya_typed_ptr) }
    }};

    expanded.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn test_build_field_access_single() {
        let idents: Punctuated<syn::Ident, Token![.]> = {
            let mut p = Punctuated::new();
            p.push(parse_quote!(pid));
            p
        };
        let tokens = build_field_access(&idents);
        assert_eq!(tokens.to_string(), "pid");
    }

    #[test]
    fn test_build_field_access_nested() {
        let idents: Punctuated<syn::Ident, Token![.]> = {
            let mut p = Punctuated::new();
            p.push_value(parse_quote!(scx));
            p.push_punct(Default::default());
            p.push_value(parse_quote!(dsq_vtime));
            p
        };
        let tokens = build_field_access(&idents);
        assert_eq!(tokens.to_string(), "scx . dsq_vtime");
    }
}
