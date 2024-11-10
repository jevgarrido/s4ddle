
# Introduction
This crate provides solvers for sparse symmetric systems of linear equations.

## Algorithms implemented
* MINRES
    - No preconditioning
    - Preconditioning through a "Black-Box" Interface

## `#![no_std]` Environments
If your application requires `#![no_std]`, make sure to disable default features. 
To do so add the following to your ```Cargo.toml```:
```toml
[dependencies.s4ddle]
version = "*"
default-features = false
```

