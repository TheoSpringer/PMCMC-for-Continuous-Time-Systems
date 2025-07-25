using SparseDiffTools, ForwardDiff, SparseArrays

n = 5
colorvec = ones(Int, n)          # trivial coloring
sparsity = sprand(Bool, n, n, 1.0)

mutable struct P
    λ::Vector{Float64}
    buf::Vector{Float64}
end
p = P(zeros(n), zeros(n))

h!(buf, z) = @. buf = z^2         # fake h(z)
f(z) = (h!(p.buf, z); dot(p.λ, p.buf))

z0 = randn(n)
cache = SparseDiffTools.ForwardAutoColorHesCache(f, z0, colorvec, sparsity)

# λ=0 → exact zero Hessian
fill!(p.λ, 0.0)
H = SparseDiffTools.autoauto_color_hessian(f, z0, cache)
@assert maximum(abs, H) == 0.0

# λ=1 → Hessian should be diagonal 2
fill!(p.λ, 1.0)
H2 = SparseDiffTools.autoauto_color_hessian(f, z0, cache)