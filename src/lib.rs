#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

pub mod embed;

#[cfg(feature = "std")]
pub mod os;

#[cfg(feature = "std")]
mod readwrite;

pub use readwrite::*;
