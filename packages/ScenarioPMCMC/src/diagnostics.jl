const THESIS_TITLE_FSIZE  = 28   # plot title
const THESIS_LABEL_FSIZE  = 22   # x/y labels (guides)
const THESIS_TICK_FSIZE   = 18   # tick labels
const THESIS_LEGEND_FSIZE = 22   # legend text
const THESIS_LW           = 5    # line width
const THESIS_MS           = 10   # marker size

using Printf: @printf
using StatsBase: Weights, sample, autocor
using Statistics: mean
using Plots
const COL_POST   = :blue       # primary: posterior / trace / signed density bars
const COL_BAND   = :lightblue  # bands / envelopes / "all predictions"
const COL_PRIOR  = :green      # prior curve
const COL_TRUE   = :red        # true value / true output
const COL_ZERO   = :black      # baseline y=0 for signed density

# Opacities + widths
const ALPHA_POST = 0.25        # posterior fill (instrumental)
const ALPHA_SIGN = 0.30        # signed density bars
const ALPHA_BAND = 0.20        # prediction envelopes / percentile bands
const ALPHA_PRI  = 0.85        # prior curve
const LW_PRI     = max(2, Int(round(0.70 * THESIS_LW)))
const LW_TRUE    = max(2, Int(round(1.20 * THESIS_LW)))
const LW_MAIN    = THESIS_LW
"""
    compute_ess(PMCMC_samples::Vector{PMCMC_sample}; max_lag::Int=100)

Compute the effective sample size (ESS) for each parameter and state.

# Arguments
- `PMCMC_samples`: PMCMC samples
- `max_lag`: maximum lag for autocorrelation estimation.

# Returns
- `ess`: vector of ESS estimates for all variables.
"""
function compute_ess(PMCMC_samples::AbstractVector; max_lag::Int=100)
    # Get number of models.
    K = size(PMCMC_samples, 1)

    # Get number of parameters of the PMCMC samples.
    n_variables = length(PMCMC_samples[1].theta) + size(PMCMC_samples[1].x_m1, 1)

    # Fill matrix with the series of the parameters of the PMCMC samples.
    sample_matrix = Array{Float64}(undef, K, n_variables)
    for i in 1:K
        # Sample initial state.
        star = sample(1:length(PMCMC_samples[i].w_m1), Weights(PMCMC_samples[i].w_m1))
        x_m1 = PMCMC_samples[i].x_m1[:, star]
        sample_matrix[i, :] .= [PMCMC_samples[i].theta; vec(x_m1)]
    end

    # Calculate the autocorrelation.
    autocorrelation = autocor(sample_matrix, Array(0:max_lag); demean=true)
    ess = zeros(n_variables)

    for i in 1:n_variables
        # Sum autocorrelation of i-th variable until first negative or max_lag.
        autocorrelation_sum = 0.0
        for lag in 1:max_lag
            if autocorrelation[lag+1, i] < 0
                break
            end
            autocorrelation_sum += autocorrelation[lag+1, i]
        end

        # Compute the effective sample size.
        ess[i] = K / (1 + 2 * autocorrelation_sum)
    end

    return ess
end

"""
    compute_gelman_rubin(PMCMC_chains::Vector{Vector{PMCMC_sample}})

Compute the Gelman–Rubin statistic for each parameter and latent state from a vector of PMCMC chains.

# Arguments
- `PMCMC_chains`: vector of chains, where each chain is a vector of PMCMC samples

# Returns
- `R_hat`: vector of R̂ values, one for each variable
"""
function compute_gelman_rubin(PMCMC_chains::AbstractVector{<:AbstractVector})
    M = length(PMCMC_chains) # Number of chains
    K = length(PMCMC_chains[1]) # Number of samples per chain

    # Get number of parameters of the PMCMC samples.
    n_variables = length(PMCMC_chains[1][1].theta) + size(PMCMC_chains[1][1].x_m1, 1)

    # Extract samples from each chain
    sample_matrices_chains = Array{Float64}[]
    for chain in PMCMC_chains
        sample_matrix = Array{Float64}(undef, K, n_variables)

        for i in 1:K
            # Sample state at the last timestep of the training dataset.
            star = sample(1:length(chain[i].w_m1), Weights(chain[i].w_m1))
            x_m1 = chain[i].x_m1[:, star]
            sample_matrix[i, :] .= [chain[i].theta; vec(x_m1)]
        end

        push!(sample_matrices_chains, sample_matrix)
    end

    # Compute means and variances
    means = zeros(M, n_variables)
    variances = zeros(M, n_variables)
    for m in 1:M
        means[m, :] .= vec(mean(sample_matrices_chains[m], dims=1))
        variances[m, :] .= vec(var(sample_matrices_chains[m], dims=1, corrected=true))
    end

    # Between-chain and within-chain variance
    mean_overall = mean(means, dims=1)
    B = K / (M - 1) .* sum((means .- mean_overall) .^ 2, dims=1)
    W = mean(variances, dims=1)

    # Estimated marginal posterior variance and R̂
    V_hat = (K - 1) / K .* W .+ B / K
    R_hat = sqrt.(V_hat ./ W)

    # Clip from below at 1.0 for numerical consistency
    return max.(R_hat, 1.0)
end

"""
    test_prediction(PMCMC_samples::Vector{PMCMC_sample}, n_x, f_theta::Function, g_theta::Function, sample_v_theta::Function, sample_w_theta::Function, u_test, y_test)

Simulate the PMCMC samples forward in time and compare the predictions to the test data.

# Arguments
- `PMCMC_samples`: PMCMC samples
- `n_x`: number of states
- `f_theta`: state transition function parametrized by theta; has inputs (theta, x, u)
- `g_theta`: measurement function parametrized by theta; has inputs (theta, x, u)
- `sample_v_theta`: function that returns N samples from the process noise distribution parametrized by theta; has input (theta, N)
- `sample_w_theta`: function that returns N samples from the measurement noise distribution parametrized by theta; has input (theta, N)
- `k_n`: each model is simulated ``k_n`` times
- `u_test`: test inputs
- `y_test`: test output
"""
function test_prediction(PMCMC_samples::AbstractVector, n_x::Int,
    f_theta::Function, g_theta::Function, sample_v_theta::Function,
    sample_w_theta::Function, k_n::Int, u_test::AbstractMatrix{<:AbstractFloat},
    y_test::AbstractMatrix{<:AbstractFloat};
    t_abs::Union{Nothing,AbstractVector{<:Real}}=nothing,
    t_cont::Union{Nothing,AbstractVector{<:Real}}=nothing,
    y_cont::Union{Nothing,AbstractMatrix{<:AbstractFloat}}=nothing)
    println("### Testing model")

    # Get number of models, etc.
    K = size(PMCMC_samples, 1)
    n_y = size(y_test, 1)
    T_test = size(y_test, 2)

    # Pre-allocate.
    x_test_sim = Array{Float64}(undef, n_x, T_test + 1, K, k_n)
    y_test_sim = Array{Float64}(undef, n_y, T_test, K, k_n)

    # Simulate models forward.
    Threads.@threads for k in 1:K
        # Get current model.
        f(x, u) = f_theta(PMCMC_samples[k].theta, x, u)
        g(x, u) = g_theta(PMCMC_samples[k].theta, x, u)
        sample_v(N) = sample_v_theta(PMCMC_samples[k].theta, N)
        sample_w(N) = sample_w_theta(PMCMC_samples[k].theta, N)

        # Simulate each model k_n times.
        for kn in 1:k_n
            # Pre-allocate.
            x_loop = Array{Float64}(undef, n_x, T_test + 1)
            y_loop = Array{Float64}(undef, n_y, T_test)

            # Sample initial state.
            star = sample(1:length(PMCMC_samples[k].w_m1), Weights(PMCMC_samples[k].w_m1))
            x_m1 = PMCMC_samples[k].x_m1[:, star]
            x_loop[:, 1] .= f(x_m1, PMCMC_samples[k].u_m1) + sample_v(1)

            # Simulate model forward.
            for t in 1:T_test
                if t >= 2
                    x_loop[:, t] .= f(x_loop[:, t-1], u_test[:, t-1]) + sample_v(1)
                end
                y_loop[:, t] .= g(x_loop[:, t], u_test[:, t]) + sample_w(1)
            end

            # Store trajectory.
            x_test_sim[:, :, k, kn] .= x_loop
            y_test_sim[:, :, k, kn] .= y_loop
        end
    end

    # Reshape.
    x_test_sim = reshape(x_test_sim, (n_x, T_test + 1, K * k_n))
    y_test_sim = reshape(y_test_sim, (n_y, T_test, K * k_n))

    println("### Testing complete")

    # Plot with absolute-time axis + optional continuous truth
    plot_predictions(y_test_sim, y_test; plot_percentiles=true,
                        t=t_abs, t_cont=t_cont, y_cont=y_cont)

    mean_rmse = sqrt(mean((y_test_sim .- repeat(y_test, 1, 1, size(y_test_sim,3))) .^ 2))
    @printf("Mean RMSE: %.2f\n", mean_rmse)
end

"""
    plot_predictions(y_pred, y_test; plot_percentiles=false, y_min=nothing, y_max=nothing)

Plot the predictions and the test data.

# Arguments
- `y_pred`: matrix containing the output predictions
- `y_test`: test output trajectory
- `plot_percentiles`: if set to true, percentiles are plotted
- `y_min`: min output to be plotted as constraint
- `y_max`: max output to be plotted as constraint
"""
function plot_predictions(
    y_pred::AbstractArray, y_test::AbstractMatrix{<:AbstractFloat};
    plot_percentiles::Bool=false,
    y_min::Union{Nothing,AbstractMatrix{<:AbstractFloat}}=nothing,
    y_max::Union{Nothing,AbstractMatrix{<:AbstractFloat}}=nothing,
    t::Union{Nothing,AbstractVector{<:Real}}=nothing,
    t_cont::Union{Nothing,AbstractVector{<:Real}}=nothing,
    y_cont::Union{Nothing,AbstractMatrix{<:AbstractFloat}}=nothing
)
    T_pred = size(y_test, 2)
    n_y    = size(y_pred, 1)
    x = isnothing(t) ? collect(0:T_pred-1) :
        (length(t) == T_pred ? t :
            throw(ArgumentError("length(t) must equal size(y_test,2) (= $T_pred)")))

    for i = 1:n_y
        y_pred_mean = mean(y_pred, dims=3)[i, :, 1]
        y_pred_max  = maximum(y_pred, dims=3)[i, :, 1]
        y_pred_min  = minimum(y_pred, dims=3)[i, :, 1]
        y_pred_09   = mapslices(v -> quantile(v, 0.9), y_pred, dims=3)[i, :, 1]
        y_pred_01   = mapslices(v -> quantile(v, 0.1), y_pred, dims=3)[i, :, 1]

        # Big + thick for thesis
        p = plot(
                x, y_pred_min;
                fillrange      = y_pred_max,
                alpha          = ALPHA_BAND,
                lc             = COL_BAND,
                label          = "all predictions",
                legend         = :topleft,
                legendfontsize = THESIS_LEGEND_FSIZE,
                guidefontsize  = THESIS_LABEL_FSIZE,
                tickfontsize   = THESIS_TICK_FSIZE,
                titlefontsize  = THESIS_TITLE_FSIZE,
                lw             = LW_MAIN,
            )

        if plot_percentiles
            plot!(p, x, y_pred_01;
                fillrange = y_pred_09,
                alpha     = ALPHA_BAND,
                lc        = COL_BAND,
                label     = "10%–90% perc.",
                lw        = LW_MAIN)
        end

        plot!(p, x, y_test[i, :];
                lc    = COL_TRUE,
                label = "true output",
                lw    = LW_MAIN)

        plot!(p, x, y_pred_mean;
                lc    = COL_POST,
                label = "mean prediction",
                lw    = LW_MAIN)

        if t_cont !== nothing && y_cont !== nothing
            @assert size(y_cont,1) == n_y
            plot!(p, t_cont, vec(y_cont[i, :]);
                lw    = LW_MAIN,
                lc    = COL_TRUE,
                label = "true y_$i (continuous)")
        end

        title!("\$y_{$i}\$: predicted output vs. true output")
        ylabel!("\$y_{$i}\$")
        xlabel!("\$t\$")
        display(p)
    end
end

function plot_autocorrelation(PMCMC_samples::AbstractVector;
                              max_lag::Int = 100,
                              signs::Union{Nothing,AbstractVector{<:Real}} = nothing,
                              burn::Int = 0,
                              state_summary::Symbol = :mean,   # :mean or :sample
                              signed_mode::Symbol = :contribution) # :none or :contribution

    K = length(PMCMC_samples)
    i0 = max(1, burn + 1)
    Kuse = K - (i0 - 1)
    @assert Kuse >= 3 "Need ≥3 samples after burn-in."

    nθ = length(PMCMC_samples[1].theta)
    nx = size(PMCMC_samples[1].x_m1, 1)
    n_variables = nθ + nx

    # Build series matrix: rows=time, cols=variables
    X = Array{Float64}(undef, Kuse, n_variables)

    for (row, i) in enumerate(i0:K)
        # state summary at last training time
        if state_summary === :mean
            w = PMCMC_samples[i].w_m1
            w = w ./ sum(w)
            x_m1 = PMCMC_samples[i].x_m1 * w
        else
            star = sample(1:length(PMCMC_samples[i].w_m1), Weights(PMCMC_samples[i].w_m1))
            x_m1 = PMCMC_samples[i].x_m1[:, star]
        end
        X[row, :] .= [PMCMC_samples[i].theta; vec(x_m1)]
    end

    # Don't go crazy with lag: large lags are just noise
    L = min(max_lag, max(5, Int(floor(Kuse/4))))
    lags = collect(0:L)

    # ACF
    if signs === nothing || signed_mode === :none
        acf_mat = autocor(X, lags; demean=true)
        title_str = "Autocorrelation"
        ylab = "ACF"
    else
        @assert length(signs) == K "signs must align with PMCMC_samples."
        s = Float64.(signs[i0:Kuse + i0 - 1])  # length Kuse
        Z = X .* reshape(s, :, 1)              # signed contributions
        acf_mat = autocor(Z, lags; demean=true)
        title_str = "Signed-contribution autocorrelation"
        ylab = "ACF of (sign × value)"
    end

    # Plot (use fewer yticks; your -1:0.1:1 + huge font makes a black blob)
    p = plot(
        legend=:topleft,
        legendfontsize=THESIS_LEGEND_FSIZE,
        guidefontsize=THESIS_LABEL_FSIZE,
        tickfontsize=THESIS_TICK_FSIZE,
        titlefontsize=THESIS_TITLE_FSIZE,
        xlabel="Lag",
        ylabel=ylab,
        title=title_str,
        yticks=-1:0.5:1
    )

    for j in 1:n_variables
        if j == 1
            plot!(lags, acf_mat[:, j], lc=COL_POST, lw=LW_MAIN, label="\$\\theta\$")
        elseif 1 < j <= nθ
            plot!(lags, acf_mat[:, j], lc=COL_POST, lw=LW_MAIN, label="")
        elseif j == nθ + 1
            plot!(lags, acf_mat[:, j], lc=COL_TRUE, lw=LW_MAIN, label="\$x(t-1)\$")
        else
            plot!(lags, acf_mat[:, j], lc=COL_TRUE, lw=LW_MAIN, label="")
        end
    end

    display(p)
    return acf_mat
end

function plot_parameter_trace(PMCMC_samples::AbstractVector;
                              signs::Union{Nothing,AbstractVector{<:Real}}=nothing,
                              burn::Int=0)

    K = length(PMCMC_samples)
    @assert K >= 1 "Need at least one PMCMC sample."

    θlen = length(PMCMC_samples[1].theta)
    xdim = size(PMCMC_samples[1].x_m1, 1)
    n_variables = θlen + xdim

    # Sample one x_m1 per iteration from PF weights at last training time
    sample_matrix = Array{Float64}(undef, K, n_variables)
    for i in 1:K
        star = sample(1:length(PMCMC_samples[i].w_m1), Weights(PMCMC_samples[i].w_m1))
        x_m1 = PMCMC_samples[i].x_m1[:, star]
        sample_matrix[i, :] .= [PMCMC_samples[i].theta; vec(x_m1)]
    end

    i0 = max(1, burn + 1)
    iters = collect(0:(K-1))
    iters_view = @view iters[i0:end]
    Smat = @view sample_matrix[i0:end, :]

    if !isnothing(signs)
        @assert length(signs) == K "signs must align with stored PMCMC_samples (same length)."
        Ssign = collect(@view signs[i0:end])
        pos = findall(>(0), Ssign)
        neg = findall(<(0), Ssign)
    else
        Ssign = nothing
        pos = Int[]
        neg = Int[]
    end

    for i in 1:n_variables
        y = vec(@view Smat[:, i])

        p = plot(
            iters_view, y;
            lw            = THESIS_LW,
            legend        = false,
            guidefontsize = THESIS_LABEL_FSIZE,
            tickfontsize  = THESIS_TICK_FSIZE,
            titlefontsize = THESIS_TITLE_FSIZE,
        )

        if !isnothing(Ssign)
            # Overlay markers where sign is + or -
            if !isempty(pos)
                scatter!(p, iters_view[pos], y[pos];
                         ms=THESIS_MS, label="sign=+")
            end
            if !isempty(neg)
                scatter!(p, iters_view[neg], y[neg];
                         ms=THESIS_MS, markershape=:x, label="sign=-")
            end
            plot!(p; legend=:topright, legendfontsize=THESIS_LEGEND_FSIZE)
        end

        if i <= θlen
            title!("Trace of \$\\theta_{$i}\$")
            ylabel!("\$\\theta_{$i}\$")
        else
            idx = i - θlen
            title!("Trace of \$x_{$idx}\$")
            ylabel!("\$x_{$idx}(t-1)\$")
        end

        xlabel!("Iteration")
        display(p)
    end

    return nothing
end

# -----------------------------------------------------------------------------
# Signed (ratio) mean helper
# -----------------------------------------------------------------------------
@inline function signed_mean_scalar(x::AbstractVector{<:Real},
                                    s::AbstractVector{<:Real})
    @assert length(x) == length(s)
    denom = sum(Float64.(s))
    if abs(denom) < 1e-12
        return mean(Float64.(x)), denom
    end
    return sum(Float64.(s) .* Float64.(x)) / denom, denom
end

# -----------------------------------------------------------------------------
# Signed histogram "density" (diagnostic): can be negative.
# IMPORTANT: This is not a PDF in general when weights/signs can be negative.
# -----------------------------------------------------------------------------
function signed_hist_pdf(x::AbstractVector{<:Real},
                         s::AbstractVector{<:Real};
                         nbins::Int=50,
                         xlim::Union{Nothing,Tuple{Real,Real}}=nothing,
                         clamp_nonneg::Bool=false)   # <-- default OFF now
    @assert length(x) == length(s) "x and signs must have the same length."

    denom = sum(Float64.(s))
    if abs(denom) < 1e-12
        @warn "Sum of signs is ~0; signed histogram is unstable. denom=$(denom)"
    end

    xmin, xmax = isnothing(xlim) ? (minimum(x), maximum(x)) : (float(xlim[1]), float(xlim[2]))
    if xmin == xmax
        xmin -= 1.0
        xmax += 1.0
    end

    edges  = collect(range(xmin, stop=xmax, length=nbins+1))
    binw   = edges[2] - edges[1]
    counts = zeros(Float64, nbins)

    @inbounds for i in eachindex(x)
        xi = float(x[i])
        if xi < edges[1] || xi > edges[end]
            continue
        end
        b = (xi == edges[end]) ? nbins : searchsortedlast(edges, xi)
        if 1 <= b <= nbins
            counts[b] += float(s[i])
        end
    end

    dens = (counts ./ denom) ./ binw   # can be negative

    if clamp_nonneg
        # Diagnostic-only: makes it *look* like a PDF, but biases shape.
        dens = max.(dens, 0.0)
        area = sum(dens) * binw
        if area > 0
            dens ./= area
        else
            @warn "All bins are zero after clamping; cannot renormalize."
        end
    end

    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    return centers, dens, edges, denom
end

# -----------------------------------------------------------------------------
# Posterior PDFs for θ and x(t=0)
# - Always plot a proper (unsigned) posterior density.
# - If signs are given: overlay signed posterior mean (ratio estimator).
# - Optionally: overlay raw signed bin "density" (can be negative) for diagnostics.
# -----------------------------------------------------------------------------
function plot_parameter_pdf(
    PMCMC_samples::AbstractVector;
    bins::Int=50,
    prior_pdf::Union{Nothing,Vector{Tuple{Vector{Float64},Vector{Float64}}}}=nothing,
    true_values::Union{Nothing,AbstractVector{<:AbstractFloat}}=nothing,
    signs::Union{Nothing,AbstractVector{<:Real}}=nothing,
    burn::Int=0,
    show_signed_density::Bool=false,  # <-- new (default off)
    clamp_nonneg::Bool=false          # <-- default OFF now
)
    K = length(PMCMC_samples)
    @assert K >= 1 "Need at least one PMCMC sample."

    θlen = length(PMCMC_samples[1].theta)
    xdim = size(PMCMC_samples[1].x_m1, 1)
    n_variables = θlen + xdim

    # Draws: (θ, x0) by sampling x0 from PF particles each iteration
    sample_matrix = Array{Float64}(undef, K, n_variables)
    for i in 1:K
        star = sample(1:length(PMCMC_samples[i].w_0), Weights(PMCMC_samples[i].w_0))
        x_0  = PMCMC_samples[i].x_0[:, star]
        sample_matrix[i, :] .= [PMCMC_samples[i].theta; vec(x_0)]
    end

    # burn handling
    i0   = max(1, burn + 1)
    Smat = @view sample_matrix[i0:end, :]

    Ssign = nothing
    if signs !== nothing
        @assert length(signs) == K "signs must align with stored PMCMC_samples."
        Ssign = @view signs[i0:end]
    end

    for i in 1:n_variables
        vals = vec(@view Smat[:, i])

        # --- Always: proper posterior density (ignoring sign) ---
        p = histogram(
            vals;
            bins = bins,
            normalize = :pdf,
            label = (Ssign === nothing ? "Posterior" : "Posterior (ignoring sign)"),
            legendfontsize = THESIS_LEGEND_FSIZE,
            guidefontsize  = THESIS_LABEL_FSIZE,
            tickfontsize   = THESIS_TICK_FSIZE,
            titlefontsize  = THESIS_TITLE_FSIZE,
        )

        # --- Signed overlay: mean only (statistically well-defined via ratio) ---
        if Ssign !== nothing
            μs, denom = signed_mean_scalar(vals, collect(Ssign))
            vline!(p, [μs]; label="Signed mean", linestyle=:dot, lw=THESIS_LW)

            # Optional: diagnostic signed bin density (can go negative)
            if show_signed_density
                centers, dens, edges, denom2 = signed_hist_pdf(vals, collect(Ssign);
                                                              nbins=bins,
                                                              clamp_nonneg=clamp_nonneg)
                binw = edges[2] - edges[1]

                plot!(p, centers, dens;
                      seriestype=:bar,
                      bar_width=binw,
                      alpha=0.20,
                      label = clamp_nonneg ? "Signed bin density (clamped; biased)" :
                                             "Signed bin density (diagnostic; can be <0)")

                hline!(p, [0.0]; lw=1, label="")  # zero baseline for signed bars
                @printf("[signed pdf diag] var %d: denom=sum(signs)=%.4e (burn=%d)\n", i, denom2, burn)
            end
        end

        # Prior overlay (if given)
        if prior_pdf !== nothing && i <= length(prior_pdf)
            xs, ds = prior_pdf[i]
            plot!(p, xs, ds; label="Prior", linestyle=:dash, lw=THESIS_LW)
        end

        # True value overlay (if given)
        if true_values !== nothing && i <= length(true_values)
            vline!(p, [true_values[i]]; label="True", lw=THESIS_LW)
        end

        # Titles/labels
        if i <= θlen
            title!("Sample PDF of \$\\theta_{$i}\$")
            xlabel!("\$\\theta_{$i}\$")
        else
            idx = i - θlen
            title!("Sample PDF of \$x_{$idx}(t=0)\$")
            xlabel!("\$x_{$idx}\$")
        end
        ylabel!("Density")

        display(p)
    end

    return nothing
end


# -----------------------------------------------------------------------------
# Signed histogram density estimator (posterior density under signed correction)
# Returns centers, dens, edges, denom, binw.
# dens integrates to 1 if denom != 0, but may be negative on some bins.
# -----------------------------------------------------------------------------
function signed_hist_density(x::AbstractVector{<:Real},
                             s::AbstractVector{<:Real};
                             nbins::Int=50,
                             xlim::Union{Nothing,Tuple{Real,Real}}=nothing)

    @assert length(x) == length(s)

    w = Float64.(s)
    denom = sum(w)

    xmin, xmax = isnothing(xlim) ? (minimum(x), maximum(x)) : (float(xlim[1]), float(xlim[2]))
    if xmin == xmax
        xmin -= 1.0
        xmax += 1.0
    end

    edges = collect(range(xmin, stop=xmax, length=nbins+1))
    binw  = edges[2] - edges[1]
    counts = zeros(Float64, nbins)

    @inbounds for i in eachindex(x)
        xi = float(x[i])
        if xi < edges[1] || xi > edges[end]
            continue
        end
        b = (xi == edges[end]) ? nbins : searchsortedlast(edges, xi)
        if 1 <= b <= nbins
            counts[b] += w[i]
        end
    end

    if abs(denom) < 1e-12
        # Unstable; still return something (caller should warn)
        dens = zeros(Float64, nbins)
    else
        dens = (counts ./ denom) ./ binw
    end

    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    return centers, dens, edges, denom, binw
end
function plot_parameter_pdf_split_instrumental(PMCMC_samples::AbstractVector;
                                               bins::Int=50,
                                               prior_pdf=nothing,
                                               true_values=nothing,
                                               burn::Int=0)

    K = length(PMCMC_samples)
    @assert K >= 1

    θlen = length(PMCMC_samples[1].theta)
    xdim = size(PMCMC_samples[1].x_m1, 1)          # <-- SPLIT state dim
    n_variables = θlen + xdim

    # sample (θ, x_split*) where x_split* ~ p(x|y_train,θ) from PF at split
    sample_matrix = Array{Float64}(undef, K, n_variables)
    for i in 1:K
        star = sample(1:length(PMCMC_samples[i].w_m1), Weights(PMCMC_samples[i].w_m1))
        xS   = PMCMC_samples[i].x_m1[:, star]       # <-- SPLIT cloud
        sample_matrix[i, :] .= [PMCMC_samples[i].theta; vec(xS)]
    end

    i0   = max(1, burn + 1)
    Smat = @view sample_matrix[i0:end, :]

    for j in 1:n_variables
        vals = vec(@view Smat[:, j])

        p = histogram(vals;
            bins=bins, normalize=:pdf,
            c=:blue, alpha=0.25, linecolor=:blue,
            label="posterior (instrumental)",
            legend=:topright,
            legendfontsize=THESIS_LEGEND_FSIZE,
            guidefontsize=THESIS_LABEL_FSIZE,
            tickfontsize=THESIS_TICK_FSIZE,
            titlefontsize=THESIS_TITLE_FSIZE,
        )

        # prior only for θ if provided (we pass only θ priors)
        if prior_pdf !== nothing && j <= length(prior_pdf)
            xs, ds = prior_pdf[j]
            plot!(p, xs, ds; lc=:green, lw=max(2, Int(round(0.70*THESIS_LW))), alpha=0.85, label="prior")
        end

        if true_values !== nothing && j <= length(true_values)
            vline!(p, [true_values[j]]; lc=:red, lw=max(2, Int(round(1.20*THESIS_LW))), label="true")
        end

        if j <= θlen
            title!(p, "Posterior of \$\\theta_{$j}\$")
            xlabel!(p, "\$\\theta_{$j}\$")
        else
            idx = j - θlen
            title!(p, "Posterior of \$x_{$idx}\$ at split (end of train)")
            xlabel!(p, "\$x_{$idx}\$")
        end
        ylabel!(p, "Density")
        display(p)
    end
    return nothing
end


function plot_parameter_pdf_split_signed_only(PMCMC_samples::AbstractVector;
                                              bins::Int=50,
                                              prior_pdf=nothing,
                                              true_values=nothing,
                                              signs::AbstractVector{<:Real},
                                              burn::Int=0)

    K = length(PMCMC_samples)
    @assert K >= 1
    @assert length(signs) == K

    θlen = length(PMCMC_samples[1].theta)
    xdim = size(PMCMC_samples[1].x_m1, 1)          # <-- SPLIT state dim
    n_variables = θlen + xdim

    # signed histogram of (θ, x_split*)
    sample_matrix = Array{Float64}(undef, K, n_variables)
    for i in 1:K
        star = sample(1:length(PMCMC_samples[i].w_m1), Weights(PMCMC_samples[i].w_m1))
        xS   = PMCMC_samples[i].x_m1[:, star]       # <-- SPLIT cloud
        sample_matrix[i, :] .= [PMCMC_samples[i].theta; vec(xS)]
    end

    i0   = max(1, burn + 1)
    Smat = @view sample_matrix[i0:end, :]
    Ssgn = Float64.(@view signs[i0:end])

    for j in 1:n_variables
        vals = vec(@view Smat[:, j])
        centers, dens, edges, denom, binw = signed_hist_density(vals, Ssgn; nbins=bins)

        p = plot(centers, dens;
            seriestype=:bar, bar_width=binw,
            c=:blue, alpha=0.30, linecolor=:blue,
            label="signed posterior density",
            legend=:topright,
            legendfontsize=THESIS_LEGEND_FSIZE,
            guidefontsize=THESIS_LABEL_FSIZE,
            tickfontsize=THESIS_TICK_FSIZE,
            titlefontsize=THESIS_TITLE_FSIZE,
        )
        hline!(p, [0.0]; lw=1, lc=:black, label="")

        if prior_pdf !== nothing && j <= length(prior_pdf)
            xs, ds = prior_pdf[j]
            plot!(p, xs, ds; lc=:green, lw=max(2, Int(round(0.70*THESIS_LW))), alpha=0.85, label="prior")
        end

        if true_values !== nothing && j <= length(true_values)
            vline!(p, [true_values[j]]; lc=:red, lw=max(2, Int(round(1.20*THESIS_LW))), label="true")
        end

        if j <= θlen
            title!(p, "Signed posterior density of \$\\theta_{$j}\$")
            xlabel!(p, "\$\\theta_{$j}\$")
        else
            idx = j - θlen
            title!(p, "Signed posterior density of \$x_{$idx}\$ at split")
            xlabel!(p, "\$x_{$idx}\$")
        end
        ylabel!(p, "Density (signed)")
        display(p)
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Signed posterior PDF plotter (θ and x0)
# -----------------------------------------------------------------------------
function plot_parameter_pdf_instrumental(PMCMC_samples::AbstractVector;
                                         bins::Int=50,
                                         prior_pdf::Union{Nothing,Vector{Tuple{Vector{Float64},Vector{Float64}}}}=nothing,
                                         true_values::Union{Nothing,AbstractVector{<:AbstractFloat}}=nothing,
                                         burn::Int=0)

    K = length(PMCMC_samples)
    @assert K >= 1

    θlen = length(PMCMC_samples[1].theta)
    xdim = size(PMCMC_samples[1].x_0, 1)
    n_variables = θlen + xdim

    # Per-iter sample (theta, x0*) where x0* is sampled from PF at that iter
    sample_matrix = Array{Float64}(undef, K, n_variables)
    for i in 1:K
        star = sample(1:length(PMCMC_samples[i].w_0), Weights(PMCMC_samples[i].w_0))
        x0s  = PMCMC_samples[i].x_0[:, star]
        sample_matrix[i, :] .= [PMCMC_samples[i].theta; vec(x0s)]
    end

    i0   = max(1, burn + 1)
    Smat = @view sample_matrix[i0:end, :]

    for j in 1:n_variables
        vals = vec(@view Smat[:, j])

        p = histogram(vals;
            bins=bins,
            normalize=:pdf,
            c=COL_POST,
            alpha=ALPHA_POST,
            linecolor=COL_POST,
            label="posterior (ignoring sign)",
            legend=:topright,
            legendfontsize=THESIS_LEGEND_FSIZE,
            guidefontsize=THESIS_LABEL_FSIZE,
            tickfontsize=THESIS_TICK_FSIZE,
            titlefontsize=THESIS_TITLE_FSIZE,
        )

        # Prior
        if prior_pdf !== nothing && j <= length(prior_pdf)
            xs, ds = prior_pdf[j]
            plot!(p, xs, ds; lc=COL_PRIOR, lw=LW_PRI, alpha=ALPHA_PRI, label="prior")
        end

        # True value
        if true_values !== nothing && j <= length(true_values)
            vline!(p, [true_values[j]]; lc=COL_TRUE, lw=LW_TRUE, label="true")
        end

        if j <= θlen
            title!(p, "Instrumental posterior of \$\\theta_{$j}\$ (ignoring sign)")
            xlabel!(p, "\$\\theta_{$j}\$")
        else
            idx = j - θlen
            title!(p, "Instrumental posterior of \$x_{$idx}(t=0)\$ (ignoring sign)")
            xlabel!(p, "\$x_{$idx}\$")
        end
        ylabel!(p, "Density")
        display(p)
    end
    return nothing
end



function plot_parameter_pdf_signed_only(PMCMC_samples::AbstractVector;
                                        bins::Int=50,
                                        prior_pdf::Union{Nothing,Vector{Tuple{Vector{Float64},Vector{Float64}}}}=nothing,
                                        true_values::Union{Nothing,AbstractVector{<:AbstractFloat}}=nothing,
                                        signs::AbstractVector{<:Real},
                                        burn::Int=0)

    K = length(PMCMC_samples)
    @assert K >= 1
    @assert length(signs) == K

    θlen = length(PMCMC_samples[1].theta)
    xdim = size(PMCMC_samples[1].x_0, 1)
    n_variables = θlen + xdim

    sample_matrix = Array{Float64}(undef, K, n_variables)
    for i in 1:K
        star = sample(1:length(PMCMC_samples[i].w_0), Weights(PMCMC_samples[i].w_0))
        x0s  = PMCMC_samples[i].x_0[:, star]
        sample_matrix[i, :] .= [PMCMC_samples[i].theta; vec(x0s)]
    end

    i0   = max(1, burn + 1)
    Smat = @view sample_matrix[i0:end, :]
    Ssgn = Float64.( @view signs[i0:end] )

    for j in 1:n_variables
        vals = vec(@view Smat[:, j])

        centers, dens, edges, denom, binw = signed_hist_density(vals, Ssgn; nbins=bins)

        p = plot(centers, dens;
            seriestype=:bar,
            bar_width=binw,
            c=COL_POST,
            alpha=ALPHA_SIGN,
            linecolor=COL_POST,
            label="signed posterior density",
            legend=:topright,
            legendfontsize=THESIS_LEGEND_FSIZE,
            guidefontsize=THESIS_LABEL_FSIZE,
            tickfontsize=THESIS_TICK_FSIZE,
            titlefontsize=THESIS_TITLE_FSIZE,
        )

        # zero baseline
        hline!(p, [0.0]; lw=1, lc=COL_ZERO, label="")

        # Prior
        if prior_pdf !== nothing && j <= length(prior_pdf)
            xs, ds = prior_pdf[j]
            plot!(p, xs, ds; lc=COL_PRIOR, lw=LW_PRI, alpha=ALPHA_PRI, label="prior")
        end

        # True value
        if true_values !== nothing && j <= length(true_values)
            vline!(p, [true_values[j]]; lc=COL_TRUE, lw=LW_TRUE, label="true")
        end

        if j <= θlen
            title!(p, "Signed posterior density of \$\\theta_{$j}\$")
            xlabel!(p, "\$\\theta_{$j}\$")
        else
            idx = j - θlen
            title!(p, "Signed posterior density of \$x_{$idx}(t=0)\$")
            xlabel!(p, "\$x_{$idx}\$")
        end
        ylabel!(p, "Density (signed)")

        if abs(denom) < 1e-8 * length(Ssgn)
            @printf("[plot_parameter_pdf_signed_only] WARNING var %d: denom=sum(signs)=%.4e small -> noisy density.\n", j, denom)
        end

        display(p)
    end
    return nothing
end


function plot_parameter_trace_signed_simplified(PMCMC_samples::AbstractVector;
                                                signs::Union{Nothing,AbstractVector{<:Real}}=nothing,
                                                burn::Int=0,
                                                show_running_signed_mean::Bool=true)

    K = length(PMCMC_samples)
    @assert K >= 1 "Need at least one PMCMC sample."

    θlen = length(PMCMC_samples[1].theta)
    xdim = size(PMCMC_samples[1].x_m1, 1)
    n_variables = θlen + xdim

    # Sample one x_m1 per iteration from PF weights (last training time)
    sample_matrix = Array{Float64}(undef, K, n_variables)
    for i in 1:K
        star = sample(1:length(PMCMC_samples[i].w_m1), Weights(PMCMC_samples[i].w_m1))
        x_m1 = PMCMC_samples[i].x_m1[:, star]
        sample_matrix[i, :] .= [PMCMC_samples[i].theta; vec(x_m1)]
    end

    i0 = max(1, burn + 1)
    iters = collect(0:(K-1))
    itv = @view iters[i0:end]
    Smat = @view sample_matrix[i0:end, :]

    Ssign = nothing
    pos = Int[]
    neg = Int[]
    if signs !== nothing
        @assert length(signs) == K "signs must align with stored PMCMC_samples."
        Ssign = Float64.( @view signs[i0:end] )
        pos = findall(>(0), Ssign)
        neg = findall(<(0), Ssign)
    end

    for j in 1:n_variables
        y = vec(@view Smat[:, j])

        p = plot(
            itv, y;
            lw=LW_MAIN,
            lc=COL_POST,
            label="trace",
            legend=:topright,
            legendfontsize=THESIS_LEGEND_FSIZE,
            guidefontsize=THESIS_LABEL_FSIZE,
            tickfontsize=THESIS_TICK_FSIZE,
            titlefontsize=THESIS_TITLE_FSIZE,
        )

        if Ssign !== nothing
            # sign markers (keep)
            if !isempty(pos)
                scatter!(p, itv[pos], y[pos];
                         ms=THESIS_MS,
                         markershape=:circle,
                         markerstrokecolor=COL_POST,
                         markercolor=:white,
                         label="sign=+")
            end
            if !isempty(neg)
                scatter!(p, itv[neg], y[neg];
                         ms=THESIS_MS,
                         markershape=:x,
                         markerstrokecolor=COL_POST,
                         label="sign=-")
            end

            # running signed mean: marker-only (no linestyle)
            if show_running_signed_mean
                num = cumsum(Ssign .* y)
                den = cumsum(Ssign)
                run = similar(num)
                @inbounds for t in eachindex(num)
                    run[t] = (abs(den[t]) < 1e-12) ? NaN : (num[t] / den[t])
                end
                scatter!(p, itv, run;
                         ms=max(4, Int(round(0.60 * THESIS_MS))),
                         markershape=:diamond,
                         markerstrokecolor=COL_POST,
                         markercolor=:white,
                         label="running signed mean")
            end
        end

        if j <= θlen
            title!(p, "Trace of \$\\theta_{$j}\$")
            ylabel!(p, "\$\\theta_{$j}\$")
        else
            idx = j - θlen
            title!(p, "Trace of \$x_{$idx}\$")
            ylabel!(p, "\$x_{$idx}(t-1)\$")
        end
        xlabel!(p, "Iteration")
        display(p)
    end
    return nothing
end
