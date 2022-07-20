module DifferentiableFactorizations

export diff_qr, diff_cholesky, diff_lu, diff_eigen, diff_svd

using LinearAlgebra, ImplicitDifferentiation, ComponentArrays, ChainRulesCore

# QR

function qr_conditions(A, x)
  (; Q, R) = x
  return vcat(
    vec(UpperTriangular(Q' * Q) + LowerTriangular(R) - I - Diagonal(R)),
    vec(Q * R - A),
  )
end
function qr_forward(A)
  qr_res = qr(A)
  Q = copy(qr_res.Q[:, 1:size(A, 2)])
  (; R) = qr_res
  return ComponentVector(; Q, R)
end
const _diff_qr = ImplicitFunction(qr_forward, qr_conditions)
function diff_qr(A)
  (; Q, R) = _diff_qr(A)
  return (; Q, R)
end

# Cholesky

function cholesky_conditions(A, U)
  return vec(UpperTriangular(U' * U) + LowerTriangular(U) - UpperTriangular(A) - Diagonal(U))
end
function cholesky_forward(A)
  ch_res = cholesky(A)
  return ch_res.U
end
const _diff_cholesky = ImplicitFunction(cholesky_forward, cholesky_conditions)
function diff_cholesky(A)
  U = _diff_cholesky(A)
  return (; L = U', U)
end

# LU

function lu_conditions(A, LU)
  (; L, U, p) = LU
  pint = convert(Vector{Int}, p)
  return vcat(
    vec(UpperTriangular(L) + LowerTriangular(U) - Diagonal(U) - I),
    vec(L * U - A[pint, :]),
    p,
  )
end
function lu_forward(A)
  lu_res = lu(A)
  (; L, U, p) = lu_res
  return ComponentVector(; L, U, p)
end
const _diff_lu = ImplicitFunction(lu_forward, lu_conditions)
function diff_lu(A)
  temp = _diff_lu(A)
  (; L, U, p) = temp
  return (; L, U, p = convert(Vector{Int}, p))
end

# Eigen

comp_vec(A) = ComponentVector((; A))
comp_vec(A, B) = ComponentVector((; A, B))
function ChainRulesCore.rrule(::typeof(comp_vec), A)
  out = comp_vec(A)
  T = typeof(out)
  return out, Δ -> begin
    _Δ = convert(T, Δ)
    (NoTangent(), _Δ.A)
  end
end
function ChainRulesCore.rrule(::typeof(comp_vec), A, B)
  out = comp_vec(A, B)
  T = typeof(out)
  return out, Δ -> begin
    _Δ = convert(T, Δ)
    (NoTangent(), _Δ.A, _Δ.B) 
  end
end

function eigen_conditions(AB, sV)
  (; s, V) = sV
  (; A) = AB
  if hasproperty(AB, :B)
    (; B) = AB
  else
    B = I
  end
  return vcat(
    vec(A * V - B * V * Diagonal(s)),
    diag(V' * B * V) .- 1,
  )
end
function eigen_forward(AB)
  (; A) = AB
  if hasproperty(AB, :B)
    (; B) = AB
    eig_res = eigen(A, B)
  else
    eig_res = eigen(A)
  end
  s = eig_res.values
  V = eig_res.vectors
  return ComponentVector(; s, V)
end

const _diff_eigen = ImplicitFunction(eigen_forward, eigen_conditions)
function diff_eigen(A)
  (; s, V) = _diff_eigen(comp_vec(A))
  return (; s , V)
end
function diff_eigen(A, B)
  (; s, V) = _diff_eigen(comp_vec(A, B))
  return (; s , V)
end

# SVD

function svd_conditions(A, USV)
  (; U, S, V) = USV
  VtV = V' * V
  return vcat(
    vec(U * Diagonal(S) * V' - A),
    vec(UpperTriangular(VtV) + LowerTriangular(U' * U) - 2I),
    diag(VtV) .- 1,
  )
end
function svd_forward(A)
  svd_res = svd(A)
  (; U, S, V) = svd_res
  return ComponentVector(; U, S, V)
end

const _diff_svd = ImplicitFunction(svd_forward, svd_conditions)
function diff_svd(A)
  (; U, S, V) = _diff_svd(A)
  return (; U, S , V)
end

end
