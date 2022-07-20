# DifferentiableFactorizations

This package contains a bunch of differentiable matrix factorizations differentiated using the implicit function theorem as implemented in [`ImplicitDifferentiation.jl`](https://github.com/gdalle/ImplicitDifferentiation.jl). The derivatives computed are only correct if the **computed** factorization of the matrix is unique and differentiable. If the implementation does not guarantee uniqueness and differentiability, the solution reported cannot be trusted. Theoretically in some cases, the factorization may only be unique up to a permutation or a sign flip. In those cases if the implementation guarantees a unique and differentiable output, the derivatives reported are still valid. This can be experimentally tested by comparing against finite difference.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/mohamed82008/DifferentiableFactorizations.jl")
Pkg.add("Zygote")
```

## Loading

```julia
using DifferentiableFactorizations, Zygote, LinearAlgebra
```

## Cholesky factorization

```julia
A = rand(3, 3)

# Forward pass: A = L * L' or A = U' * U (i.e. L' == U)
(; L, U) = diff_cholesky(A' * A + 2I)

# Differentiation
f(A) = diff_cholesky(A' * A + 2I).L
zjac = Zygote.jacobian(f, A)[1]
```

## LU factorization

```julia
A = rand(3, 3)

# Forward pass: A[p, :] = L * U for a permutation vector p
(; L, U, p) = diff_lu(A)

# Differentiation
f(A) = vec(diff_lu(A).U)
zjac = Zygote.jacobian(f, A)[1]
```

## QR factorization

```julia
A = rand(3, 2)

# Forward pass: A = Q * R
(; Q, R) = diff_qr(A)

# Differentiation
f(A) = vec(diff_qr(A).Q)
zjac = Zygote.jacobian(f, A)[1]
```

## Singular value decomposition (SVD)

```julia
A = rand(3, 2)

# Forward pass: A = U * Diagonal(S) * V'
(; U, S, V) = diff_svd(A)

# Differentiation
f(A) = vec(diff_svd(A).U)
zjac = Zygote.jacobian(f, A)[1]
```

## Eigenvalue decomposition

```julia
A = rand(3, 3)

# Forward pass: `s` are the eigenvalues and `V` are the eigenvectors
(; s, V) = diff_eigen(A' * A)

# Differentiation
f(A) = vec(diff_eigen(A' * A).s)
zjac = Zygote.jacobian(f, A)[1]
```

## Generalized eigenvalue decomposition

```julia
A = rand(3, 3)
B = rand(3, 3)

# Forward pass: `s` are the eigenvalues and `V` are the eigenvectors
(; s, V) = diff_eigen(A' * A, B' * B + 2I)

# Differentiation
f(B) = vec(diff_eigen(A' * A, B' * B + 2I).V)
zjac = Zygote.jacobian(f, B)[1]
```
