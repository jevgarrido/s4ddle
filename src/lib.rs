#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

pub mod operators;

pub use minres::*;
pub use minres_black_box::*;
pub use minres_black_box_precond::*;

mod minres;
mod minres_black_box;
mod minres_black_box_precond;

#[cfg(feature = "std")]
pub mod readwrite;
