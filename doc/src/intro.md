
# Introduction

This crate provides a variety of solvers for linear systems of equations.
In this context, several methods are implemented and an attempt is made at providing several different interfaces to such methods.

One major reason to write several interfaces is enabling the execution of the methods in 'no\_std' environments as well as in environments where the standard library is available, for this, two modules, ```embed``` and ```os```, are provided which tackle these environment requirements respectively.

Generally speaking, different systems and applications have different needs in terms of what data types can be used, the ammount of memory available, ability for parallel execution, wether or not preconditioning is necessary, etc. .


## Methods Provided

* CR
* MINRES
* SYMMLQ

## 'no\_std' Environments
If your application needs a 'no\_std' compatible linear solver, choose the interfaces available in module ```embed```. 



## 'std' Environments
