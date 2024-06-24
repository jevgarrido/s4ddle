
# Introduction
This crate provides solvers for sparse symmetric systems of linear equations in the following form:
$$ (A - s I ) x = b $$
$$ M (A - s I ) x = M b $$

Where \\(A\\) is a Real symmetric coefficient matrix (no assumption on its definiteness),
\\(s\\) is a Real scalar value,
\\(I\\) is the identity matrix,
\\(b\\) is an arbitrary Real vector,
\\(M\\) is a Real full-rank square preconditioning matrix (no assumption on its symmetry).

Internally, the preconditioned case is treated as follows:
$$ M (A - s I ) M^\top y = M b \, \quad x = M^\top y $$
which constitutes a symmetric system in terms of \\(y\\).

## Methods implemented
* CR
    - [ ] No preconditioning
    - [ ] Preconditioning through a "Black-Box" Interface
* SYMMLQ
    - [ ] No preconditioning
    - [ ] Preconditioning through a "Black-Box" Interface
* MINRES
    - [x] No preconditioning
    - [ ] Preconditioning through a "Black-Box" Interface
* MINRES_QLP 
    - [ ] No preconditioning
    - [ ] Preconditioning through a "Black-Box" Interface

MINRES_QLP returns the pseudoinverse solution if \\(A\\) is singular. However, each iteration of MINRES_QLP is more expensive that that of MINRES.

## `#![no_std]` Environments
If your application requires `#![no_std]`, make sure to disable default features. 
To do so add the following to your ```Cargo.toml```:
```toml
[dependencies.s4ddle]
version = "*"
default-features = false
```

