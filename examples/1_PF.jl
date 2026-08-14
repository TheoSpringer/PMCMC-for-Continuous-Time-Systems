using Random
using Distributions: Geometric, MvNormal
using LinearAlgebra
using StatsBase: Weights, sample
import ScenarioBase: PMCMC_sample
using Statistics: mean, cov
using Printf: @printf

# =============================================================================
# NUMERICAL UTILITIES (log-sum-exp + signed-log arithmetic)
# =============================================================================
@inline _nw() = size(G, 2)

@inline function _logsumexp(v::AbstractVector{<:Real})
    m = maximum(v)
    if !isfinite(m); return -Inf; end
    return m + log(sum(exp.(v .- m)))
end

@inline function _logaddexp(a::Real, b::Real)
    m = max(a, b)
    if m == -Inf; return -Inf; end
    return m + log(exp(a - m) + exp(b - m))
end

# log(1 - exp(x)) for x <= 0, computed stably
@inline function _log1mexp(x::Real)
    if x == 0
        return -Inf
    elseif x < -0.6931471805599453 # log(0.5)
        return log1p(-exp(x))
    else
        return log(-expm1(x))
    end
end

# Signed-log add: returns (sgn, logabs) for A + B, where A = sgnA*exp(logabsA)
@inline function _signed_add(sgnA::Int, logabsA::Real, sgnB::Int, logabsB::Real)
    if sgnA == 0 || logabsA == -Inf
        return (sgnB, float(logabsB))
    end
    if sgnB == 0 || logabsB == -Inf
        return (sgnA, float(logabsA))
    end
    if sgnA == sgnB
        return (sgnA, _logaddexp(logabsA, logabsB))
    end
    if logabsA == logabsB
        return (0, -Inf)
    elseif logabsA > logabsB
        return (sgnA, logabsA + _log1mexp(logabsB - logabsA))
    else
        return (sgnB, logabsB + _log1mexp(logabsA - logabsB))
    end
end

function cap_cov!(Σ; smin=1e-10, smax=0.15^2)
    # force exact symmetry first (also fixes any drift from Σ updates)
    Σ .= (Σ .+ Σ') ./ 2

    F = eigen(Symmetric(Σ))
    λ = clamp.(F.values, smin, smax)
    Σ .= F.vectors * Diagonal(λ) * F.vectors'

    # force exact symmetry again (critical for MvNormal/PDMat -> cholesky)
    Σ .= (Σ .+ Σ') ./ 2
    return Σ
end

# Returns (sgn, logabs) of exp(logZa) - exp(logZb)
@inline function _signed_logdiffexp(logZa::Real, logZb::Real)
    if logZa == -Inf && logZb == -Inf
        return (0, -Inf)
    end
    if logZa == logZb
        return (0, -Inf)
    elseif logZa > logZb
        if logZb == -Inf
            return (1, float(logZa))
        end
        return (1, logZa + _log1mexp(logZb - logZa))
    else
        if logZa == -Inf
            return (-1, float(logZb))
        end
        return (-1, logZb + _log1mexp(logZa - logZb))
    end
end

# =============================================================================
# EULER–MARUYAMA (levels: M = 2^ℓ substeps per measurement gap)
# =============================================================================
# dx(t) = f(x(t),u(t),θ) dt + G dW(t), with Q = G*G'
# EM update: x <- x + drift(x,u,θ)*h + G*sqrt(h)*ξ, ξ~N(0,I)
@inline function em_step!(x::AbstractVector,
                          u_on_gap::AbstractVector,
                          θ::AbstractVector,
                          h::Real,
                          ξ::AbstractVector,
                          b::AbstractVector)
    @inbounds begin
        drift!(b, x, u_on_gap, θ)
        x .+= b .* h .+ (G * sqrt(h)) * ξ
        if !all(isfinite, x)
            x .= NaN
        end
    end
    return x
end
# Backward-compatible wrapper (NOT used in the hot PF loops after this patch)
@inline function em_step!(x::AbstractVector, u_on_gap::AbstractVector, θ::AbstractVector, h::Real, ξ::AbstractVector)
    b = Vector{Float64}(undef, length(x))
    return em_step!(x, u_on_gap, θ, h, ξ, b)
end

@inline function simulate_level_inplace!(x::AbstractVector,
                                        u_on_gap::AbstractVector,
                                        θ::AbstractVector,
                                        Δt::Real,
                                        ℓ::Integer,
                                        rng::AbstractRNG,
                                        ξ::AbstractVector,
                                        b::AbstractVector)
    if Δt <= 0
        return x
    end
    M = 1 << ℓ
    h = Δt / M
    @inbounds for _ in 1:M
        randn!(rng, ξ)
        em_step!(x, u_on_gap, θ, h, ξ, b)
    end
    return x
end
@inline function simulate_level_inplace!(x::AbstractVector,
                                        u_on_gap::AbstractVector,
                                        θ::AbstractVector,
                                        Δt::Real,
                                        ℓ::Integer,
                                        rng::AbstractRNG,
                                        ξ::AbstractVector)
    b = Vector{Float64}(undef, length(x))
    return simulate_level_inplace!(x, u_on_gap, θ, Δt, ℓ, rng, ξ, b)
end

@inline function simulate_coupled_levels_inplace!(xf::AbstractVector,
                                                 xc::AbstractVector,
                                                 u_on_gap::AbstractVector,
                                                 θ::AbstractVector,
                                                 Δt::Real,
                                                 ℓ::Integer,
                                                 rng::AbstractRNG,
                                                 ξ1::AbstractVector,
                                                 ξ2::AbstractVector,
                                                 ξc::AbstractVector,
                                                 b::AbstractVector)
    @assert ℓ >= 1
    if Δt <= 0
        return xf, xc
    end
    M_c = 1 << (ℓ - 1)
    h_c = Δt / M_c
    invsqrt2 = inv(sqrt(2.0))

    @inbounds for _ in 1:M_c
        # fine: two half steps
        randn!(rng, ξ1)
        em_step!(xf, u_on_gap, θ, h_c/2, ξ1, b)

        randn!(rng, ξ2)
        em_step!(xf, u_on_gap, θ, h_c/2, ξ2, b)

        # coarse: coupled increment
        @inbounds @simd for k in eachindex(ξc)
            ξc[k] = (ξ1[k] + ξ2[k]) * invsqrt2
        end
        em_step!(xc, u_on_gap, θ, h_c, ξc, b)
    end
    return xf, xc
end

# Backward-compatible wrapper
@inline function simulate_coupled_levels_inplace!(xf::AbstractVector,
                                                 xc::AbstractVector,
                                                 u_on_gap::AbstractVector,
                                                 θ::AbstractVector,
                                                 Δt::Real,
                                                 ℓ::Integer,
                                                 rng::AbstractRNG,
                                                 ξ1::AbstractVector,
                                                 ξ2::AbstractVector,
                                                 ξc::AbstractVector)
    b = Vector{Float64}(undef, length(xf))
    return simulate_coupled_levels_inplace!(xf, xc, u_on_gap, θ, Δt, ℓ, rng, ξ1, ξ2, ξc, b)
end

# level-ℓ EM propagation over one measurement gap Δt with M = 2^ℓ substeps
function simulate_level!(x0::AbstractVector, u_on_gap::AbstractVector, θ::AbstractVector, Δt::Real, ℓ::Integer, rng::AbstractRNG)
    @assert ℓ >= 0
    if Δt <= 0
        return copy(x0)
    end
    M = 1 << ℓ
    h = Δt / M
    x = copy(x0)
    ξ = Vector{Float64}(undef, _nw())
    b = Vector{Float64}(undef, length(x))
    @inbounds for _ in 1:M
        randn!(rng, ξ)
        em_step!(x, u_on_gap, θ, h, ξ, b)
    end
    return x
end

# Coupled fine/coarse over one gap:
# coarse step size h_c = Δt / 2^(ℓ-1), fine uses two half steps with ξ1, ξ2
# ξc = (ξ1+ξ2)/√2 so that sqrt(h_c)*ξc matches the sum of two fine increments.
function simulate_coupled_levels_general!(xf0::AbstractVector, xc0::AbstractVector,
                                         u_on_gap::AbstractVector, θ::AbstractVector,
                                         Δt::Real, ℓ::Integer, rng::AbstractRNG)
    @assert ℓ >= 1
    if Δt <= 0
        return copy(xf0), copy(xc0)
    end
    M_c = 1 << (ℓ - 1)
    h_c = Δt / M_c
    xf = copy(xf0)
    xc = copy(xc0)
    ξ1 = Vector{Float64}(undef, _nw())
    ξ2 = Vector{Float64}(undef, _nw())
    ξc = Vector{Float64}(undef, _nw())
    b  = Vector{Float64}(undef, length(xf))
    invsqrt2 = inv(sqrt(2.0))

    @inbounds for _ in 1:M_c
        randn!(rng, ξ1)
        em_step!(xf, u_on_gap, θ, h_c/2, ξ1, b)

        randn!(rng, ξ2)
        em_step!(xf, u_on_gap, θ, h_c/2, ξ2, b)

        @inbounds @simd for k in eachindex(ξc)
            ξc[k] = (ξ1[k] + ξ2[k]) * invsqrt2
        end
        em_step!(xc, u_on_gap, θ, h_c, ξc, b)
    end
    return xf, xc
end

function em_propagate_gap!(X::AbstractMatrix, u_left::AbstractVector, θ::AbstractVector, Δt::Real;
                           h_target::Real=0.02, add_noise::Bool=true,
                           rng::AbstractRNG=Random.default_rng())
    if Δt <= 0
        return X
    end
    n_x_local, Np = size(X)
    M = max(1, ceil(Int, Δt / h_target))
    h = Δt / M

    n_w = _nw()
    b   = Vector{Float64}(undef, n_x_local)          # drift buffer
    Gh  = add_noise ? (G * sqrt(h)) : nothing

    # (keeps your current structure; biggest win here is avoiding drift allocations)
    @inbounds for _ in 1:M
        ξ = add_noise ? randn(rng, n_w, Np) : nothing
        @inbounds for j in 1:Np
            xj = @view X[:, j]

            drift!(b, xj, u_left, θ)
            if add_noise
                xj .+= b .* h .+ (Gh * @view ξ[:, j])
            else
                xj .+= b .* h
            end

            if !all(isfinite, xj)
                xj .= NaN
            end
        end
    end
    return X
end


function particle_filter_em_dt(u_aug::AbstractMatrix{<:AbstractFloat}, u_meas::AbstractMatrix{<:AbstractFloat}, y::AbstractMatrix{<:AbstractFloat}, n_x::Int, N::Int, sample_x_0::Function, θ::AbstractVector; h_target::Real, rng::AbstractRNG = Random.default_rng())
    T = size(y, 2)
    w = Array{Float64}(undef, T, N)
    x_pf = Array{Float64}(undef, n_x, T, N)
    X = _sample_x0_matrix(sample_x_0, N) # n_x × N
    logZ = 0.0
    wprev = fill(1.0/N, N)
    logw = Vector{Float64}(undef, N)
    @inbounds for t in 1:T
        ucol_t = @view u_aug[:, t]
        u_prop = u_left_from_ucol(ucol_t, model.n_u)
        Δt = max(Δt_from_ucol(ucol_t, model.n_u), 0.0)
        u_obs = @view u_meas[:, t]
        # resample
        if t >= 2
            a = sample(1:N, Weights(wprev), N)
            X = copy(@view X[:, a])
        end
        # propagate (in-place per column)
        @inbounds for j in 1:N
            xj = @view X[:, j]
            Xj = reshape(xj, :, 1) # view-backed 1-col matrix
            em_propagate_gap!(Xj, u_prop, θ, Δt; h_target=h_target, add_noise=true, rng=rng)
        end
        if !all(isfinite, X)
            # optional: fill to avoid uninitialized reads downstream
            w[t, :] .= 1.0 / N
            @views x_pf[:, t, :] .= X
            for tt in (t+1):T
                w[tt, :] .= 1.0 / N
                @views x_pf[:, tt, :] .= X
            end
            return x_pf, w, -Inf
        end
        @views x_pf[:, t, :] .= X
        # log-weights
        @inbounds for j in 1:N
            logw[j] = log_measure_density(@view(y[:, t]), @view(X[:, j]), u_obs, θ)
        end
        lse = _logsumexp(logw)
        if !isfinite(lse)
            # true failure (NaNs/Inf in dynamics or residuals)
            w[t, :] .= 1.0 / N
            for tt in (t+1):T
                w[tt, :] .= 1.0 / N
                @views x_pf[:, tt, :] .= X
            end
            return x_pf, w, -Inf
        end
        logZ += (lse - log(N))
        @inbounds for j in 1:N
            w[t, j] = exp(logw[j] - lse)  # safe normalization
        end
        wprev .= @view w[t, :]
    end
    return x_pf, w, logZ
end

# =============================================================================
# MEASUREMENT LOG-DENSITY (up to constants)
# =============================================================================
# log p(y_k | x_k, θ) = -1/2 (y - g(x,u,θ))' R^{-1} (y - g(x,u,θ))
function log_measure_density(y_k::AbstractVector, xk::AbstractVector, u_for_meas::AbstractVector, θ::AbstractVector)
    if (model.n_y == 1) && (R isa Diagonal) && (length(R.diag) == 1)
        μ  = g_theta(θ, xk, u_for_meas)
        μ1 = (μ isa Number) ? float(μ) : float(μ[1])
        r  = float(y_k[1]) - μ1
        return -0.5 * (r*r) / R.diag[1]
    else
        μ = g_theta(θ, xk, u_for_meas)
        r = y_k .- μ
        return -0.5 * (r' * (R \ r))[]  # scalar
    end
end

# Convenience: density (linear scale), useful for legacy utilities
measure_density(y_k, xk, u_for_meas, θ) = exp(log_measure_density(y_k, xk, u_for_meas, θ))

# =============================================================================
# COUPLED RESAMPLING (maximal coupling of categorical distributions)
# =============================================================================
@inline function _cat1(p::AbstractVector{<:Real}, rng::AbstractRNG)
    u = rand(rng)
    s = 0.0
    @inbounds for i in eachindex(p)
        s += p[i]
        if u <= s
            return i
        end
    end
    return lastindex(p)
end

function maximal_coupling_resample(wf::AbstractVector{<:Real}, wc::AbstractVector{<:Real}, N::Int, rng::AbstractRNG)
    @assert length(wf) == length(wc)
    M = length(wf)
    sf = sum(wf); sc = sum(wc)
    pf = sf > 0 ? wf ./ sf : fill(1.0/M, M)
    pc = sc > 0 ? wc ./ sc : fill(1.0/M, M)

    pmin = similar(pf)
    @inbounds for i in 1:M
        pmin[i] = min(pf[i], pc[i])
    end
    C = sum(pmin)
    if C <= 0
        idxf = [_cat1(pf, rng) for _ in 1:N]
        idxc = [_cat1(pc, rng) for _ in 1:N]
        return idxf, idxc
    end

    rf = pf .- pmin
    rc = pc .- pmin
    srf = sum(rf); src = sum(rc)
    rf .= (srf > 0) ? (rf ./ srf) : fill(1.0/M, M)
    rc .= (src > 0) ? (rc ./ src) : fill(1.0/M, M)
    pmin ./= C

    idxf = Vector{Int}(undef, N)
    idxc = Vector{Int}(undef, N)
    @inbounds for n in 1:N
        if rand(rng) <= C
            i = _cat1(pmin, rng)
            idxf[n] = i
            idxc[n] = i
        else
            idxf[n] = _cat1(rf, rng)
            idxc[n] = _cat1(rc, rng)
        end
    end
    return idxf, idxc
end
@inline function ess(w::AbstractVector{<:Real})
    # assumes w are normalized
    return 1.0 / max(sum(abs2, w), eps())
end

function systematic_resample(w::AbstractVector{<:Real}, rng::AbstractRNG)
    N = length(w)
    idx = Vector{Int}(undef, N)
    u0 = rand(rng) / N
    c = w[1]
    i = 1
    @inbounds for n in 1:N
        u = u0 + (n-1)/N
        while u > c && i < N
            i += 1
            c += w[i]
        end
        idx[n] = i
    end
    return idx
end

function coupled_systematic_resample(wf::AbstractVector{<:Real},
                                    wc::AbstractVector{<:Real},
                                    rng::AbstractRNG)
    # same systematic uniforms => strong coupling when wf≈wc
    N = length(wf)
    idxf = Vector{Int}(undef, N)
    idxc = Vector{Int}(undef, N)

    u0 = rand(rng) / N

    cf = wf[1]; if_ = 1
    cc = wc[1]; ic_ = 1

    @inbounds for n in 1:N
        u = u0 + (n-1)/N

        while u > cf && if_ < N
            if_ += 1
            cf += wf[if_]
        end
        while u > cc && ic_ < N
            ic_ += 1
            cc += wc[ic_]
        end

        idxf[n] = if_
        idxc[n] = ic_
    end
    return idxf, idxc
end

function gather_cols!(Xout::AbstractMatrix, Xin::AbstractMatrix, idx::AbstractVector{Int})
    @inbounds for j in 1:length(idx)
        Xout[:, j] .= @view Xin[:, idx[j]]
    end
    return Xout
end
# =============================================================================
# Parsing u_aug: [u; Δt]
# =============================================================================
u_left_from_ucol(ucol::AbstractVector, n_u::Int) = @view ucol[1:n_u]
Δt_from_ucol(ucol::AbstractVector, n_u::Int) = ucol[n_u + 1]

# =============================================================================
# PF AT ONE LEVEL ℓ: returns log(Ẑ_ℓ^{PF})
# =============================================================================
# sample_x_0 may be implemented either as:
# sample_x_0(N)::Matrix (n_x × N), OR
# sample_x_0()::Vector (n_x)
# This helper supports both.
function _sample_x0_matrix(sample_x_0::Function, N::Int)
    try
        X = sample_x_0(N)
        return X
    catch
        # fallback: call N times and stack
        cols = [sample_x_0() for _ in 1:N]
        return hcat(cols...)
    end
end

function particle_filter_level_dt(u_aug::AbstractMatrix{<:AbstractFloat},
                                  u_meas::AbstractMatrix{<:AbstractFloat},
                                  y::AbstractMatrix{<:AbstractFloat},
                                  n_x::Int, N::Int, sample_x_0::Function,
                                  θ::AbstractVector, ℓ::Integer;
                                  resample_ess_frac::Float64 = 0.7,
                                  rng::AbstractRNG = Random.default_rng())
    T = size(y, 2)
    w = Array{Float64}(undef, T, N)
    x_pf = Array{Float64}(undef, n_x, T, N)

    X = _sample_x0_matrix(sample_x_0, N) # n_x × N
    Xtmp = similar(X)

    logZ  = 0.0
    wprev = fill(1.0/N, N)

    ξ    = Vector{Float64}(undef, _nw())
    b    = Vector{Float64}(undef, n_x)     # drift buffer reused across all steps
    logw = Vector{Float64}(undef, N)

    @inbounds for t in 1:T
        ucol_t = @view u_aug[:, t]
        u_prop = u_left_from_ucol(ucol_t, model.n_u)
        Δt     = max(Δt_from_ucol(ucol_t, model.n_u), 0.0)
        u_obs  = @view u_meas[:, t]

        # resample (only if ESS low)
        if t >= 2
            if ess(wprev) < resample_ess_frac * N
                a = sample(1:N, Weights(wprev), N)
                gather_cols!(Xtmp, X, a)
                X, Xtmp = Xtmp, X
            end
        end

        # propagate at level ℓ (buffered)
        @inbounds for j in 1:N
            xj = @view X[:, j]
            simulate_level_inplace!(xj, u_prop, θ, Δt, ℓ, rng, ξ, b)
        end

        if !all(isfinite, X)
            w[t, :] .= 1.0 / N
            @views x_pf[:, t, :] .= X
            for tt in (t+1):T
                w[tt, :] .= 1.0 / N
                @views x_pf[:, tt, :] .= X
            end
            return x_pf, w, -Inf
        end
        @views x_pf[:, t, :] .= X

        # weights
        @inbounds for j in 1:N
            logw[j] = log_measure_density(@view(y[:, t]), @view(X[:, j]), u_obs, θ)
        end
        lse = _logsumexp(logw)
        if !isfinite(lse)
            logZ = -Inf
            @inbounds begin
                w[t, :] .= 1.0 / N
                for tt in (t+1):T
                    w[tt, :] .= 1.0 / N
                    @views x_pf[:, tt, :] .= X
                end
            end
            return x_pf, w, -Inf
        end

        logZ += (lse - log(N))
        @inbounds for j in 1:N
            w[t, j] = exp(logw[j] - lse)
        end
        wprev .= @view w[t, :]
    end

    return x_pf, w, logZ
end

# overload: if u_meas omitted, default to left input part of u_aug
particle_filter_level_dt(u_aug::AbstractMatrix{<:AbstractFloat}, y::AbstractMatrix{<:AbstractFloat}, n_x::Int, N::Int, sample_x_0::Function, θ::AbstractVector, ℓ::Integer; kwargs...) =
    particle_filter_level_dt(u_aug, @view(u_aug[1:model.n_u, :]), y, n_x, N, sample_x_0, θ, ℓ; kwargs...)

# =============================================================================
# COUPLED PF (ℓ, ℓ-1): returns (log Ẑ_ℓ^{PF}, log Ẑ_{ℓ-1}^{PF})
# Coupling is:
# - identical initial particles
# - maximal coupling resampling
# - Brownian coupling inside each gap (fine/coarse EM)
# =============================================================================
function particle_filter_coupled_dt(u_aug::AbstractMatrix{<:AbstractFloat},
                                    u_meas::AbstractMatrix{<:AbstractFloat},
                                    y::AbstractMatrix{<:AbstractFloat},
                                    n_x::Int, N::Int, sample_x_0::Function,
                                    θ::AbstractVector, ℓ::Integer;
                                    resample_ess_frac::Float64 = 0.7,
                                    rng::AbstractRNG = Random.default_rng())
    @assert ℓ >= 1
    T = size(y, 2)

    Xf = _sample_x0_matrix(sample_x_0, N)
    Xc = copy(Xf)

    Xf_tmp = similar(Xf)
    Xc_tmp = similar(Xc)

    logZf = 0.0
    logZc = 0.0

    wfprev = fill(1.0/N, N)
    wcprev = fill(1.0/N, N)

    ξ1 = Vector{Float64}(undef, _nw())
    ξ2 = Vector{Float64}(undef, _nw())
    ξc = Vector{Float64}(undef, _nw())
    b  = Vector{Float64}(undef, n_x)      # drift buffer reused

    logwf = Vector{Float64}(undef, N)
    logwc = Vector{Float64}(undef, N)

    @inbounds for t in 1:T
        ucol_t = @view u_aug[:, t]
        u_prop = u_left_from_ucol(ucol_t, model.n_u)
        Δt     = max(Δt_from_ucol(ucol_t, model.n_u), 0.0)
        u_obs  = @view u_meas[:, t]

        if t >= 2
            if min(ess(wfprev), ess(wcprev)) < resample_ess_frac * N
                idxf, idxc = maximal_coupling_resample(wfprev, wcprev, N, rng)
                gather_cols!(Xf_tmp, Xf, idxf); Xf, Xf_tmp = Xf_tmp, Xf
                gather_cols!(Xc_tmp, Xc, idxc); Xc, Xc_tmp = Xc_tmp, Xc
            end
        end

        # coupled propagate (buffered)
        @inbounds for j in 1:N
            xf = @view Xf[:, j]
            xc = @view Xc[:, j]
            simulate_coupled_levels_inplace!(xf, xc, u_prop, θ, Δt, ℓ, rng, ξ1, ξ2, ξc, b)
        end

        @inbounds for j in 1:N
            logwf[j] = log_measure_density(@view(y[:, t]), @view(Xf[:, j]), u_obs, θ)
            logwc[j] = log_measure_density(@view(y[:, t]), @view(Xc[:, j]), u_obs, θ)
        end

        lsef = _logsumexp(logwf)
        if !isfinite(lsef)
            return -Inf, -Inf
        end
        logZf += (lsef - log(N))
        @inbounds for j in 1:N
            wfprev[j] = exp(logwf[j] - lsef)
        end

        lsec = _logsumexp(logwc)
        if !isfinite(lsec)
            return -Inf, -Inf
        end
        logZc += (lsec - log(N))
        @inbounds for j in 1:N
            wcprev[j] = exp(logwc[j] - lsec)
        end
    end

    return logZf, logZc
end

# overload: if u_meas omitted
particle_filter_coupled_dt(u_aug::AbstractMatrix{<:AbstractFloat}, y::AbstractMatrix{<:AbstractFloat}, n_x::Int, N::Int, sample_x_0::Function, θ::AbstractVector, ℓ::Integer; kwargs...) =
    particle_filter_coupled_dt(u_aug, @view(u_aug[1:model.n_u, :]), y, n_x, N, sample_x_0, θ, ℓ; kwargs...)

# =============================================================================
# MLPF unbiased likelihood estimator (Rhee–Glynn Russian roulette on PF Ẑ_ℓ)
# Returns: x_pf0, w0, log|Ẑ_MLPF|, sign(Ẑ_MLPF)
# We form:
# Ẑ_MLPF = Ẑ_{ℓ0} + Σ_{ℓ=ℓ0+1}^N (Ẑ_ℓ - Ẑ_{ℓ-1}) / P(N ≥ ℓ)
# with N = ℓ0 + Geometric(1-ρ), so P(N ≥ ℓ) = ρ^(ℓ-ℓ0).
# IMPORTANT: Ẑ_MLPF can be negative (telescoping sum). That breaks standard PMMH
# unless you implement a signed pseudo-marginal correction.
# =============================================================================
@inline function _spawn_rng(rng::AbstractRNG)
    return MersenneTwister(rand(rng, UInt))
end

function _mlpf_single(u_aug::AbstractMatrix{<:AbstractFloat},
                      u_meas::AbstractMatrix{<:AbstractFloat},
                      y::AbstractMatrix{<:AbstractFloat},
                      n_x::Int, N::Int, sample_x_0::Function,
                      θ::AbstractVector;
                      ℓ0::Integer, ρ::Float64, ℓ_max::Integer,
                      resample_ess_frac::Float64,
                      rng::AbstractRNG)

    # Base level ℓ0 PF
    x_pf0, w0, logZ0 = particle_filter_level_dt(u_aug, u_meas, y, n_x, N, sample_x_0, θ, ℓ0;
                                                resample_ess_frac=resample_ess_frac, rng=rng)

    S_sgn::Int = 1
    S_logabs::Float64 = float(logZ0)

    Nlvl = ℓ0 + rand(rng, Geometric(1 - ρ))
    Nlvl = min(Nlvl, ℓ_max)

    for ℓ in (ℓ0+1):Nlvl
        logZf, logZc = particle_filter_coupled_dt(u_aug, u_meas, y, n_x, N, sample_x_0, θ, ℓ;
                                                  resample_ess_frac=resample_ess_frac, rng=rng)
        Δ_sgn, Δ_logabs = _signed_logdiffexp(logZf, logZc)
        if Δ_sgn != 0
            term_sgn = Δ_sgn
            term_logabs = Δ_logabs - (ℓ - ℓ0) * log(ρ)
            S_sgn, S_logabs = _signed_add(S_sgn, S_logabs, term_sgn, term_logabs)
        end
    end

    return x_pf0, w0, S_sgn, S_logabs
end

function mlpf_unbiased_likelihood(u_aug::AbstractMatrix{<:AbstractFloat},
                                 u_meas::AbstractMatrix{<:AbstractFloat},
                                 y::AbstractMatrix{<:AbstractFloat},
                                 n_x::Int, N::Int, sample_x_0::Function,
                                 θ::AbstractVector;
                                 ℓ0::Integer = 1,
                                 ρ::Float64 = 0.6,
                                 ℓ_max::Integer = ℓ0 + 6,
                                 resample_ess_frac::Float64 = 0.7,
                                 n_rep_mlpf::Int = 2,
                                 rng::AbstractRNG = Random.default_rng())

    @assert 0.0 < ρ < 1.0
    @assert ℓ0 >= 0
    @assert n_rep_mlpf >= 1

    # First replicate provides x_pf0,w0 for diagnostics
    rng1 = _spawn_rng(rng)
    x_pf0, w0, sgn1, logabs1 = _mlpf_single(u_aug, u_meas, y, n_x, N, sample_x_0, θ;
                                            ℓ0=ℓ0, ρ=ρ, ℓ_max=ℓ_max,
                                            resample_ess_frac=resample_ess_frac,
                                            rng=rng1)

    # Accumulate replicates in signed-log domain
    S_sgn = sgn1
    S_logabs = logabs1

    for r in 2:n_rep_mlpf
        rngr = _spawn_rng(rng)
        _, _, sgnr, logabsr = _mlpf_single(u_aug, u_meas, y, n_x, N, sample_x_0, θ;
                                           ℓ0=ℓ0, ρ=ρ, ℓ_max=ℓ_max,
                                           resample_ess_frac=resample_ess_frac,
                                           rng=rngr)
        S_sgn, S_logabs = _signed_add(S_sgn, S_logabs, sgnr, logabsr)
    end

    # average => subtract log(n_rep_mlpf)
    if S_sgn == 0 || S_logabs == -Inf
        return x_pf0, w0, -Inf, 0.0
    end
    S_logabs -= log(n_rep_mlpf)
    return x_pf0, w0, S_logabs, float(S_sgn)
end

# overload: if u_meas omitted
mlpf_unbiased_likelihood(u_aug::AbstractMatrix{<:AbstractFloat}, y::AbstractMatrix{<:AbstractFloat}, n_x::Int, N::Int, sample_x_0::Function, θ::AbstractVector; kwargs...) =
    mlpf_unbiased_likelihood(u_aug, @view(u_aug[1:model.n_u, :]), y, n_x, N, sample_x_0, θ; kwargs...)

# =============================================================================
# LEGACY: per-gap unbiased increment (kept for comparison / debugging)
# =============================================================================
function unbiased_increment(y_k::AbstractVector, x_prev::AbstractVector, u_on_gap::AbstractVector, θ::AbstractVector, Δt::Real; K::Integer=1, ρ::Float64=0.4, rng::AbstractRNG=Random.default_rng())
    @assert 0.0 < ρ < 1.0
    @assert K ≥ 0
    xK = simulate_level!(x_prev, u_on_gap, θ, Δt, K, rng)
    Lhat = measure_density(y_k, xK, u_on_gap, θ)
    x_last = xK
    Nlvl = K + rand(rng, Geometric(1 - ρ))
    for ℓ in (K+1):Nlvl
        xf, xc = simulate_coupled_levels_general!(x_prev, x_prev, u_on_gap, θ, Δt, ℓ, rng)
        Δℓ = measure_density(y_k, xf, u_on_gap, θ) - measure_density(y_k, xc, u_on_gap, θ)
        Lhat += Δℓ / (ρ^(ℓ - K))
        x_last = xf
    end
    return (Lhat, x_last)
end

function unbiased_increment_from_ucol(y_k::AbstractVector, x_prev::AbstractVector, ucol::AbstractVector, θ::AbstractVector; K::Integer=1, ρ::Float64=0.4, rng::AbstractRNG=Random.default_rng())
    uL = u_left_from_ucol(ucol, model.n_u)
    Δt = Δt_from_ucol(ucol, model.n_u)
    return unbiased_increment(y_k, x_prev, uL, θ, max(Δt, 0.0); K=K, ρ=ρ, rng=rng)
end

# =============================================================================
# GENERIC PF WRAPPER (used by :em_coarse / :em_fine baselines)
# =============================================================================
function particle_filter_dt(u_aug::AbstractMatrix{<:AbstractFloat}, u_meas::AbstractMatrix{<:AbstractFloat}, y::AbstractMatrix{<:AbstractFloat}, n_x::Int, N::Int, sample_x_0::Function, per_gap_step_and_weight::Function)
    T = size(y, 2)
    w = Array{Float64}(undef, T, N)
    x_pf = Array{Float64}(undef, n_x, T, N)
    a = Array{Int}(undef, T, N)
    # init
    for n in 1:N
        x_pf[:, 1, n] = try
            sample_x_0()
        catch
            # fallback if only sample_x_0(N) exists
            _sample_x0_matrix(sample_x_0, N)[:, n]
        end
    end

    logZ = 0.0
    for t in 1:T
        ucol_t = @view u_aug[:, t]
        u_obs = @view u_meas[:, t]
        if t >= 2
            a[t, :] .= sample(1:N, Weights(@view w[t-1, :]), N)
        else
            a[t, :] .= 1:N
        end
        for n in 1:N
            x_prev = @view x_pf[:, t == 1 ? 1 : t-1, a[t, n]]
            w_inc, xnew = per_gap_step_and_weight(@view(y[:, t]), x_prev, ucol_t, u_obs)
            w[t, n] = w_inc
            x_pf[:, t, n] = xnew
        end
        s = sum(@view w[t, :])
        if !(isfinite(s) && s > 0.0)
            logZ = -Inf
            # fill remainder safely
            w[t, :] .= 1.0 / N
            @inbounds for tt in (t+1):T
                w[tt, :] .= 1.0 / N
                for j in 1:N
                    x_pf[:, tt, j] .= x_pf[:, t, j]
                end
            end
            return x_pf, w, -Inf
        end
        w[t, :] ./= s
        logZ += log(s) - log(N)
    end
    return x_pf, w, logZ
end

particle_filter_dt(u_aug, y, n_x, N, sample_x_0, per_gap_step_and_weight) =
    particle_filter_dt(u_aug, @view(u_aug[1:model.n_u, :]), y, n_x, N, sample_x_0, per_gap_step_and_weight)

# =============================================================================
# PMMH (proper MH chain): store current state every stored iteration
# - for :mlpf_signed we accept using log|Ẑ| and keep sign for post-processing
# =============================================================================
function particle_MMH_dt(u_aug::AbstractMatrix{<:AbstractFloat}, u_meas::AbstractMatrix{<:AbstractFloat}, y::AbstractMatrix{<:AbstractFloat}, n_x::Int, K::Int, K_b::Int, k_d::Int, N::Int, g_theta::Function, sample_x_0::Function, log_pdf_w_theta::Function, log_pdf_theta::Function, propose_theta::Function, log_ratio_proposal_pdf::Function, theta_init::AbstractVector{<:AbstractFloat};
    print_progress::Bool=true, ℓ0::Integer=1, ρ_tail::Float64=0.3, ℓ_max::Integer = ℓ0 + 6, resample_ess_frac::Float64 = 0.7, n_rep_mlpf::Int = 2,variant::Symbol=:mlpf_signed, h_coarse::Real=0.20, h_fine::Real=0.02, rng::AbstractRNG = Random.default_rng())

    K_total = K_b + 1 + (K - 1) * (k_d + 1)
    n_theta = length(theta_init)
    n_u = model.n_u

    PMMH_samples = [PMCMC_sample(zeros(n_theta), Array{Float64}(undef, n_x, N), zeros(N), zeros(n_u), Array{Float64}(undef, n_x, N), zeros(N)) for _ in 1:K]

    # initialise at θ_init
    θ = copy(theta_init)
    logpθ = log_pdf_theta(θ); logpθ = (logpθ isa Number) ? logpθ : logpθ[]
    x_pf_cur = nothing
    w_cur = nothing
    loglike_abs_cur = -Inf
    sgn_like_cur = 1.0
    signs_kept = Float64[]

    if variant === :mlpf_signed
        x_pf_cur, w_cur, loglike_abs_cur, sgn_like_cur =
            mlpf_unbiased_likelihood(u_aug, u_meas, y, n_x, N, sample_x_0, θ;
                             ℓ0=ℓ0, ρ=ρ_tail, ℓ_max=ℓ_max,
                             resample_ess_frac=resample_ess_frac,
                             n_rep_mlpf=n_rep_mlpf,
                             rng=rng)
    elseif variant === :em_coarse || variant === :em_fine
        h_target_here = (variant === :em_coarse) ? h_coarse : h_fine
        x_pf_cur, w_cur, logZ =
            particle_filter_em_dt(u_aug, u_meas, y, n_x, N, sample_x_0, θ; h_target=h_target_here, rng=rng)
        loglike_abs_cur = logZ
        sgn_like_cur = 1.0
    else
        error("Unknown variant: $(variant)")
    end

    accepted = 0
    current = 1

    n_pf_eval = 0
    n_pf_bad = 0
    n_logalpha_bad = 0

    if print_progress
        @printf("PMMH progress: %3d%% | iter %d/%d | accept %d | acc-rate %.1f%%", 0, 0, K_total, 0, 0.0)
        flush(stdout)
    end

    for iter in 1:K_total
        θprop = propose_theta(θ)
        logpθprop = log_pdf_theta(θprop); logpθprop = (logpθprop isa Number) ? logpθprop : logpθprop[]
        if isfinite(logpθprop)
            local x_pf_prop, w_prop, loglike_abs_prop, sgn_prop
            if variant === :mlpf_signed
                x_pf_prop, w_prop, loglike_abs_prop, sgn_prop =
                    mlpf_unbiased_likelihood(u_aug, u_meas, y, n_x, N, sample_x_0, θprop;
                             ℓ0=ℓ0, ρ=ρ_tail, ℓ_max=ℓ_max,
                             resample_ess_frac=resample_ess_frac,
                             n_rep_mlpf=n_rep_mlpf,
                             rng=rng)
            else
                h_target_here = (variant === :em_coarse) ? h_coarse : h_fine
                per_gap = function (y_k, x_prev, ucol_t, u_obs)
                    u_prop = u_left_from_ucol(ucol_t, model.n_u)
                    Δt = max(Δt_from_ucol(ucol_t, model.n_u), 0.0)
                    X1 = reshape(copy(x_prev), :, 1)
                    em_propagate_gap!(X1, u_prop, θprop, Δt; h_target=h_target_here, add_noise=true, rng=rng)
                    xnew = vec(X1)
                    winc = exp(log_measure_density(y_k, xnew, u_obs, θprop))
                    return (winc, xnew)
                end
                x_pf_prop, w_prop, logZprop =
                    particle_filter_dt(u_aug, u_meas, y, n_x, N, sample_x_0, per_gap)
                loglike_abs_prop = logZprop
                sgn_prop = 1.0
            end

            n_pf_eval += 1
            if !isfinite(loglike_abs_prop)
                n_pf_bad += 1
            end

            # accept using log|Ẑ| (signed pseudo-marginal)
            logα = (loglike_abs_prop - loglike_abs_cur) + (logpθprop - logpθ) + log_ratio_proposal_pdf(θ, θprop)
            if isnan(logα)
                n_logalpha_bad += 1
            else
                if log(rand(rng)) < logα
                    θ = θprop
                    logpθ = logpθprop
                    x_pf_cur = x_pf_prop
                    w_cur = w_prop
                    loglike_abs_cur = loglike_abs_prop
                    sgn_like_cur = float(sgn_prop)
                    accepted += 1
                end
            end
        end

        # store after burn-in/thinning (store current even if rejected)
        if (iter > K_b) && (mod(iter - (K_b + 1), k_d + 1) == 0)
            @views begin
                PMMH_samples[current].theta .= θ
                PMMH_samples[current].x_m1 .= x_pf_cur[:, end, :]
                PMMH_samples[current].w_m1 .= w_cur[end, :]
                PMMH_samples[current].u_m1 .= u_aug[1:model.n_u, end]
                PMMH_samples[current].x_0 .= x_pf_cur[:, 1, :]
                PMMH_samples[current].w_0 .= w_cur[1, :]
            end
            push!(signs_kept, float(sgn_like_cur))
            current += 1
        end

        if print_progress
            pct = floor(Int, 100 * iter / K_total)
            acc = 100 * accepted / iter
            @printf("\rPMMH progress: %3d%% | iter %d/%d | accept %d | acc-rate %.1f%%", pct, iter, K_total, accepted, acc)
            flush(stdout)
        end
    end

    if print_progress
        println()
    end
    @printf("\n[PF diagnostics] ran PF %d times | bad PF %d (%.1f%%) | nonfinite logα %d\n",
        n_pf_eval, n_pf_bad, 100*n_pf_bad/max(n_pf_eval,1), n_logalpha_bad)

    return PMMH_samples, (100 * accepted / K_total), signs_kept
end

# overload: if u_meas omitted
particle_MMH_dt(u_aug, y, n_x, K, K_b, k_d, N, g_theta, sample_x_0, log_pdf_w_theta, log_pdf_theta, propose_theta, log_ratio_proposal_pdf, theta_init; kwargs...) =
    particle_MMH_dt(u_aug, @view(u_aug[1:model.n_u, :]), y, n_x, K, K_b, k_d, N, g_theta, sample_x_0, log_pdf_w_theta, log_pdf_theta, propose_theta, log_ratio_proposal_pdf, theta_init; kwargs...)

# =============================================================================
# STAGED PMMH (passes u_meas through; keeps signs per stage)
# =============================================================================
# ---------------------------------------------------------------------------
# Stage-specific burn-in schedule (cheap early, ramps up, full burn-in at end)
# Paste this helper ABOVE staged_PMMH_dt (anywhere in 1_PF.jl before the function)
# ---------------------------------------------------------------------------
@inline function stage_burnin(i::Int, Nst::Int, K_b_final::Int, K_stage::Int)
    # If there is only one stage, it is the final stage.
    if Nst <= 1 || i == Nst
        return K_b_final
    end

    # Pre-final stages: keep burn-in small and ramp it up with stage index.
    # Cap by min(K_stage, 100) so staging stays cheap (and never exceeds what you keep).
    max_pre = min(max(10, K_stage), 100)

    # i=1 => x=0 => 0 burn-in
    # i≈Nst-1 => x≈1 => ~max_pre burn-in
    x = (i - 1) / (Nst - 1)   # in [0,1)
    return Int(round(max_pre * x^2))  # convex ramp: very small early, more later
end


# ---------------------------------------------------------------------------
# REPLACE your entire staged_PMMH_dt with the version below (copy/paste)
# ---------------------------------------------------------------------------
function staged_PMMH_dt(u_aug::AbstractMatrix{<:AbstractFloat}, u_meas::AbstractMatrix{<:AbstractFloat}, y::AbstractMatrix{<:AbstractFloat},
                        n_x::Int, K::Int, K_b::Int, k_d::Int, N::Int,
                        g_theta::Function, sample_x_0::Function, log_pdf_w_theta::Function, log_pdf_theta::Function,
                        theta_init::AbstractVector{<:AbstractFloat}, proposal_cov_init::AbstractMatrix{<:AbstractFloat},
                        T_chunk::Int, K_stage::Int, alpha;
                        Q::AbstractMatrix, print_progress::Bool=true,
                        regularizer::Real=1e-8, ℓ0::Integer=1, ρ_tail::Float64=0.3, ℓ_max::Integer = ℓ0 + 6,
                        resample_ess_frac::Float64 = 0.7, n_rep_mlpf::Int = 2,
                        variant::Symbol=:mlpf_signed, h_coarse::Real=0.20, h_fine::Real=0.02,
                        rng::AbstractRNG = Random.default_rng())

    n_theta = length(theta_init)
    T = size(y, 2)
    N_stages = ceil(Int, T / T_chunk)

    θ = copy(theta_init)
    Σ = copy(proposal_cov_init)
    acceptance_ratio = zeros(N_stages)
    PMMH_samples = Vector{PMCMC_sample}(undef, K)
    all_signs = Vector{Float64}[]

    if print_progress
        println("### Started staged PMMH (Δt-aware PF)")
    end

    for i in 1:N_stages
        T_i  = min(i * T_chunk, T)
        u_i  = u_aug[:,  1:T_i]
        um_i = u_meas[:, 1:T_i]
        y_i  = y[:,      1:T_i]

        if print_progress
            @printf("\n-- Stage %d/%d (1..%d points) --\n", i, N_stages, T_i)
        end

        propose_theta(θcur) = reflect_box!(rand(MvNormal(θcur, Symmetric(Σ))), θ_lo, θ_hi)
        log_ratio_proposal_pdf(θacc, θprop) = 0.0

        # ---- Particle schedule: NEVER force a floor ABOVE the user-chosen N ----
        N_final = N
        N_min_stage   = min(N_final, max(100, round(Int, 0.3 * N_final)))
        N_adapt_floor = min(N_final, max(150, round(Int, 0.5 * N_final)))

        N_i_raw = round(Int, N_final * (T_i / T))
        if i < N_stages
            N_i = max(N_adapt_floor, N_i_raw)
        else
            N_i = max(N_min_stage, N_i_raw)  # here N_i_raw == N_final
        end

        # ---- NEW: stage-specific burn-in ----
        # Early stages: very little burn-in; later stages: more; final stage: full K_b.
        Kb_i = stage_burnin(i, N_stages, K_b, K_stage)

        local PMMH_stage, acc, signs_stage
        if i < N_stages
            PMMH_stage, acc, signs_stage =
                particle_MMH_dt(u_i, um_i, y_i, n_x,
                                K_stage, Kb_i, 0, N_i,     # <-- changed K_b -> Kb_i
                                g_theta, sample_x_0, log_pdf_w_theta, log_pdf_theta,
                                propose_theta, log_ratio_proposal_pdf, θ;
                                print_progress=true,
                                ℓ0=ℓ0, ρ_tail=ρ_tail, ℓ_max=ℓ_max,
                                resample_ess_frac=resample_ess_frac,
                                n_rep_mlpf=n_rep_mlpf,
                                variant=variant, h_coarse=h_coarse, h_fine=h_fine,
                                rng=rng)
        else
            # Final stage (full data): full run + full burn-in + thinning as requested
            PMMH_stage, acc, signs_stage =
                particle_MMH_dt(u_i, um_i, y_i, n_x,
                                K, K_b, k_d, N_i,          # <-- full burn-in here
                                g_theta, sample_x_0, log_pdf_w_theta, log_pdf_theta,
                                propose_theta, log_ratio_proposal_pdf, θ;
                                print_progress=true,
                                ℓ0=ℓ0, ρ_tail=ρ_tail, ℓ_max=ℓ_max,
                                resample_ess_frac=resample_ess_frac,
                                n_rep_mlpf=n_rep_mlpf,
                                variant=variant, h_coarse=h_coarse, h_fine=h_fine,
                                rng=rng)
        end

        acceptance_ratio[i] = acc
        push!(all_signs, signs_stage)

        # ---- Stage-to-stage covariance adaptation ----
        if i < N_stages
            Theta = hcat([s.theta for s in PMMH_stage]...)
            post_cov = cov(permutedims(Theta))
            Σ_new = alpha * post_cov + regularizer * Matrix(I, n_theta, n_theta)
            Σ = 0.8 * Σ + 0.2 * Σ_new

            # If acceptance collapses, shrink proposal (free improvement; no runtime cost)
            if acc < 0.10
                Σ .*= 0.5
            end

            diag_pre = diag(Σ)
            cap_cov!(Σ; smax=0.15^2)
            diag_post = diag(Σ)

            capped = any(abs.(diag_post .- diag_pre) .> 1e-14) ||
                     any(diag_post .>= 0.15^2 * 0.999)

            @printf("[stage %d] T_i=%d  N_i=%d  Kb_i=%d  acc=%.3f  capped=%s\n",
                    i, T_i, N_i, Kb_i, acc, string(capped))
            @printf("          diag(Σ) pre-cap = [% .4g, % .4g, % .4g]\n",
                    diag_pre[1], diag_pre[2], diag_pre[3])
            @printf("          diag(Σ) post-cap= [% .4g, % .4g, % .4g]\n",
                    diag_post[1], diag_post[2], diag_post[3])

            θ = Theta[:, end]
        else
            PMMH_samples = PMMH_stage
        end
    end

    if print_progress
        @printf("### Staged PMMH complete\nAverage acceptance ratio: %.2f %%\n", mean(acceptance_ratio))
    end
    return PMMH_samples, acceptance_ratio, all_signs
end



# overload: if u_meas omitted
staged_PMMH_dt(u_aug, y, n_x, K, K_b, k_d, N, g_theta, sample_x_0, log_pdf_w_theta, log_pdf_theta, theta_init, proposal_cov_init, T_chunk, K_stage, alpha; kwargs...) =
    staged_PMMH_dt(u_aug, @view(u_aug[1:model.n_u, :]), y, n_x, K, K_b, k_d, N, g_theta, sample_x_0, log_pdf_w_theta, log_pdf_theta,
                   theta_init, proposal_cov_init, T_chunk, K_stage, alpha; kwargs...)

# =============================================================================
# SIGNED PSEUDO-MARGINAL POST-PROCESSING
# =============================================================================
extract_final_signs(all_signs) = isempty(all_signs) ? Float64[] : all_signs[end]

function signed_expectation(samples::AbstractVector, signs::AbstractVector, f::Function; burn::Int=0)
    @assert length(samples) == length(signs) "signs must align with stored PMMH_samples."
    idx0 = max(1, burn + 1)
    denom = sum(@view signs[idx0:end])
    if abs(denom) < 1e-10
        @warn "Sum of signs is ~0; signed estimator unstable. Increase N_pf / improve coupling / adjust ρ or ℓ0. denom=$(denom)"
    end
    num = nothing
    for m in idx0:length(samples)
        val = f(samples[m].theta)
        if num === nothing
            num = signs[m] .* val
        else
            num .+= signs[m] .* val
        end
    end
    est = num ./ denom
    S = @view signs[idx0:end]
    meanS = mean(S)
    fracNeg = mean(S .< 0)
    Ms = length(S)
    ess_sign = (sum(S)^2) / max(sum(S.^2), eps())
    info = (; meanS, fracNeg, ess_sign, denom, Ms)
    return est, denom, info
end

function signed_mean_theta(PMMH_samples::AbstractVector, signs_kept::AbstractVector; burn::Int=0)
    μθ, denom, info = signed_expectation(PMMH_samples, signs_kept, θ -> θ; burn=burn)
    return μθ, denom, info
end

function signed_cov_theta(PMMH_samples::AbstractVector, signs_kept::AbstractVector; burn::Int=0)
    μθ, denom, info = signed_mean_theta(PMMH_samples, signs_kept; burn=burn)
    idx0 = max(1, burn + 1)
    nθ = length(PMMH_samples[idx0].theta)
    Σ = zeros(nθ, nθ)
    for m in idx0:length(PMMH_samples)
        d = PMMH_samples[m].theta .- μθ
        Σ .+= signs_kept[m] .* (d * d')
    end
    Σ ./= denom
    return Σ, μθ, info
end

# =============================================================================
# END 1_PF.jl
# =============================================================================
