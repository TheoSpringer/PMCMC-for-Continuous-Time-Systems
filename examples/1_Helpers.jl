# -----------------------------------------------------------------------------
# PURPOSE
#   Reusable utilities used by the simulation:
#     - random input generator (correlated AR(1))
#     - ZOH interpolators
#     - ODE wrappers (continuous_dynamics!, rollout_from_aug)
#     - tiny helpers for plotting bands & summarizing arrays
# =============================================================================

using LinearAlgebra
using Random
using Distributions
using DifferentialEquations
using Plots
using Random

# ─────────────────────────────────────────────────────────────────────────────
# 1) Correlated random inputs (AR(1) per channel)
# Generate random inputs at arbitrary times t_meas (ZOH between them).
# ρ_base is the correlation per unit time; effective ρ_k = exp(log(ρ_base) * Δt_k).
# ─────────────────────────────────────────────────────────────────────────────
function make_random_U_at_times(t_meas; seed=12, μ=2.0, σ=0.1,
                                ρ_base=0.95, bounds=(0.0, 50.0), n_u::Int)
    T = length(t_meas)
    @assert issorted(t_meas)
    rng = MersenneTwister(seed)

    μv = μ isa AbstractVector ? μ : fill(μ, n_u)
    σv = σ isa AbstractVector ? σ : fill(σ, n_u)

    U = zeros(n_u, T)
    ξ = randn(rng, n_u)  # noise buffer

    # init
    U[:, 1] .= clamp.(μv .+ σv .* ξ, bounds[1], bounds[2])

    for k in 2:T
        Δt = t_meas[k] - t_meas[k-1]
        ρΔ = ρ_base ^ Δt
        s  = sqrt(max(0.0, 1 - ρΔ^2))

        # new noise per column
        ξ .= randn(rng, n_u)

        U[:, k] .= clamp.( μv .+ ρΔ .* (U[:, k-1] .- μv) .+ (σv .* s) .* ξ,
                            bounds[1], bounds[2] )
    end
    return U
end

# Make parameters inside bound
function reflect_box!(x::AbstractVector, lo::AbstractVector, hi::AbstractVector)
    @inbounds for i in eachindex(x)
        L, H = lo[i], hi[i]

        if isfinite(L) && isfinite(H)
            while x[i] < L || x[i] > H
                if x[i] < L
                    x[i] = 2L - x[i]
                else
                    x[i] = 2H - x[i]
                end
            end

        elseif isfinite(L)  # lower bound only
            while x[i] < L
                x[i] = 2L - x[i]
            end
            # keep strictly inside if you want:
            if x[i] == L
                x[i] += eps(L)
            end

        elseif isfinite(H)  # upper bound only
            while x[i] > H
                x[i] = 2H - x[i]
            end
            if x[i] == H
                x[i] -= eps(H)
            end
        end
    end
    return x
end


# ─────────────────────────────────────────────────────────────────────────────
# 2) ZOH interpolators: used to calculate u_of_t_meas - Input constant beween measurements
# ─────────────────────────────────────────────────────────────────────────────
function zoh_interpolator(t_points::AbstractVector{<:Real},
                            U_values::AbstractMatrix{<:Real})
    @assert issorted(t_points) "t_points must be sorted in increasing order"
    return t -> begin
        idx = searchsortedlast(t_points, t)
        idx = clamp(idx, 1, size(U_values, 2))
        @view U_values[:, idx]          # returns a view of size (n_u, 1)
    end
end


function _unique_sorted_tol(v::Vector{Float64}; tol::Float64=1e-12)
    out = Float64[]
    for x in v
        if isempty(out) || abs(x - out[end]) > tol
            push!(out, x)
        end
    end
    return out
end

"""
    simulate_truth_em(t_meas, U_meas, θ_true, x0;
                      dt_truth=H_TRUTH, rng=MersenneTwister(2025))

Simulate ONE Euler–Maruyama SDE path with max step dt_truth using ZOH input from (t_meas,U_meas).
Returns:
  t_dense, X_dense, Y_dense  : dense trajectory (step dt_truth, includes t_end exactly)
  X_meas,  Y_meas            : states/outputs exactly at measurement times t_meas (same path)
Notes:
- Uses left-hold input u(t_prev) for propagation over each small step.
- Output at time t uses u(t) (ZOH gives u_k at t_k).
"""
function simulate_truth_em(t_meas::Vector{<:Real},
                           U_meas::AbstractMatrix{<:Real},
                           θ_true::AbstractVector,
                           x0::AbstractVector;
                           dt_truth::Real=H_TRUTH,
                           rng::AbstractRNG=MersenneTwister(2025),
                           Q_truth::Union{Nothing,AbstractMatrix}=nothing,
                           add_process_noise::Bool=true)

    @assert issorted(t_meas)
    t0 = Float64(first(t_meas))
    t1 = Float64(last(t_meas))

    # ZOH input function (returns u_k for t in [t_k, t_{k+1}))
    u_of_t = zoh_interpolator(collect(Float64, t_meas), U_meas)

    # dense grid (ensure exact endpoint)
    t_dense = collect(range(t0, stop=t1, step=dt_truth))
    if abs(t_dense[end] - t1) > 1e-12
        push!(t_dense, t1)
    end

    # union grid: ensures every step length <= dt_truth and includes all measurement times
    t_all = sort(vcat(t_dense, collect(Float64, t_meas)))
    t_all = _unique_sorted_tol(collect(Float64, t_all); tol=1e-12)

    n_x = length(x0)
    n_y = model.n_y

    X_dense = Matrix{Float64}(undef, n_x, length(t_dense))
    Y_dense = Matrix{Float64}(undef, n_y, length(t_dense))
    X_meas  = Matrix{Float64}(undef, n_x, length(t_meas))
    Y_meas  = Matrix{Float64}(undef, n_y, length(t_meas))

    # indices for recording
    id = 1
    im = 1

    x = copy(x0)
    tcur = t_all[1]

    # record helper
    function record!(t_now::Float64)
        # dense
        if id <= length(t_dense) && abs(t_now - t_dense[id]) <= 1e-12
            X_dense[:, id] .= x
            Y_dense[:, id] .= model.output(x, u_of_t(t_now), (theta=θ_true,), 0.0)
            id += 1
        end
        # measurement times
        if im <= length(t_meas) && abs(t_now - Float64(t_meas[im])) <= 1e-12
            X_meas[:, im] .= x
            @views Y_meas[:, im] .= model.output(x, U_meas[:, im], (theta=θ_true,), 0.0)
            im += 1
        end
    end

    record!(tcur)

    # --- LOCAL process noise for truth ONLY ---
    # If Q_truth is provided, use it; otherwise:
    #   add_process_noise=true  -> use global G
    #   add_process_noise=false -> set diffusion to zero
    n_w = size(G, 2)
    ξ = Vector{Float64}(undef, n_w)

    # local diffusion factor for truth
    local_G = if Q_truth === nothing
        add_process_noise ? G : zeros(size(G))
    else
        add_process_noise ? cholesky(Q_truth).L : zeros(n_x, size(Q_truth, 1))
    end

    for k in 2:length(t_all)
        tnext = t_all[k]
        h = tnext - tcur
        if h > 0
            uL = u_of_t(tcur)  # left-hold
            b = model.dynamics(x, uL, (theta=θ_true,), 0.0)

            if add_process_noise
                randn!(rng, ξ)
                x .+= b .* h .+ (local_G * sqrt(h)) * ξ
            else
                x .+= b .* h
            end
        end
        tcur = tnext
        record!(tcur)
    end

    @assert im == length(t_meas) + 1
    @assert id == length(t_dense) + 1

    return t_dense, X_dense, Y_dense, X_meas, Y_meas
end

# ─────────────────────────────────────────────────────────────────────────────
# 4) Tiny helpers used by plotting/post-processing
# ─────────────────────────────────────────────────────────────────────────────
mean3(A) = dropdims(mean(A; dims=3), dims=3)
qfun(A, p) = dropdims(mapslices(t -> quantile(t, p), A; dims=3), dims=3)

# shaded-band helper (kept here so plots stay clean in the script)
band!(p, t, lo, hi; lbl="90% band", α=0.22, col=:cyan) = plot!(
    p, t, lo; fillrange=hi, label=lbl, linealpha=0.0, fillalpha=α, color=col, fillcolor=col)