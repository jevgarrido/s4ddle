#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

pub mod operators;

pub use minres::*;

mod minres;

#[cfg(feature = "std")]
pub mod readwrite;
