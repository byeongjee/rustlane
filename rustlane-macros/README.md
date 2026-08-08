# rustlane-macros

Procedural macro implementation for
[`rustlane`](https://crates.io/crates/rustlane), an ISPC-style SPMD programming
model over Rust's portable SIMD API.

This crate provides the `#[kernel]` and `#[export]` attributes, the
`#[derive(SpmdValue)]` derive, and the kernel-world function-like macros used by
`rustlane`. The generated code targets APIs in the `rustlane` runtime, so the
two crates are released together.

Most users should depend only on `rustlane`, which re-exports these macros:

```toml
[dependencies]
rustlane = "0.1.1"
```

The facade crate requires nightly Rust because its runtime is built on
`std::simd`. See the [project README](https://github.com/byeongjee/rustlane#readme)
for usage, examples, and current limitations.

## License

Licensed under the [MIT License](LICENSE).
