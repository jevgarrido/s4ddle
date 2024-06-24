#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

pub mod operators;

pub use cr::*;
pub use minres::*;
pub use symmlq::*;
pub use minres_qlp::*;

mod cr;
mod minres;
mod symmlq;
mod minres_qlp;

#[cfg(feature = "std")]
pub mod readwrite;
