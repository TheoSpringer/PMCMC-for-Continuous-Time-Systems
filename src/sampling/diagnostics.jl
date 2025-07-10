"""
    compute_ess(PMCMC_samples::Vector{PMCMC_sample}; max_lag::Int=100)

Compute the effective sample size (ESS) for each parameter and state.

# Arguments
- `PMCMC_samples`: PMCMC samples
- `max_lag`: maximum lag for autocorrelation estimation.

# Returns
- `ess`: vector of ESS estimates for all variables.
"""
function compute_ess(PMCMC_samples::Vector{PMCMC_sample}; max_lag::Int=100)
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
function compute_gelman_rubin(PMCMC_chains::Vector{Vector{PMCMC_sample}})
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
- `u_test`: test input
- `y_test`: test output
"""
function test_prediction(PMCMC_samples::Vector{PMCMC_sample}, n_x::Int, f_theta::Function, g_theta::Function, sample_v_theta::Function, sample_w_theta::Function, k_n::Int, u_test::AbstractMatrix{<:AbstractFloat}, y_test::AbstractMatrix{<:AbstractFloat})
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

    # Plot results.
    plot_predictions(y_test_sim, y_test; plot_percentiles=true)

    # Compute and print RMSE.
    mean_rmse = sqrt(mean((y_test_sim .- repeat(y_test, 1, 1, K * k_n)) .^ 2))
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
function plot_predictions(y_pred::AbstractArray, y_test::AbstractMatrix{<:AbstractFloat}; plot_percentiles::Bool=false, y_min::Union{Nothing,AbstractMatrix{<:AbstractFloat}}=nothing, y_max::Union{Nothing,AbstractMatrix{<:AbstractFloat}}=nothing)
    # Get prediction horizon and number of outputs.
    T_pred = size(y_test, 2)
    n_y = size(y_pred, 1)

    # Plot the predictions and the test data for all output dimensions.
    for i = 1:n_y
        # Calculate median, mean, maximum, and minimum prediction.
        y_pred_med = median(y_pred, dims=3)[i, :, 1]
        y_pred_mean = mean(y_pred, dims=3)[i, :, 1]

        y_pred_max = maximum(y_pred, dims=3)[i, :, 1]
        y_pred_min = minimum(y_pred, dims=3)[i, :, 1]

        # Calculate percentiles.
        y_pred_09 = mapslices(x -> quantile(x, 0.9), y_pred, dims=3)[i, :, 1]
        y_pred_01 = mapslices(x -> quantile(x, 0.1), y_pred, dims=3)[i, :, 1]

        # Plot range of predictions.
        p = plot(Array(0:T_pred-1), y_pred_min, fillrange=y_pred_max, alpha=0.35, label="all predictions", legend=:topleft)

        # Plot percentiles.
        if plot_percentiles
            plot!(Array(0:T_pred-1), y_pred_01, fillrange=y_pred_09, alpha=0.35, label="10% perc. - 90% perc.")
        end

        # Plot true output.
        plot!(Array(0:T_pred-1), y_test[i, :], label="true output", lw=2)

        # Plot median/mean prediction.
        # plot!(Array(0:T_pred-1), y_pred_med, label="median prediction", lw=2)
        plot!(Array(0:T_pred-1), y_pred_mean, label="mean prediction", lw=2)

        # Plot constraints.
        if y_min !== nothing
            plot!(Array(0:T_pred-1), y_min[i, :], fillrange=minimum([y_pred_min; y_test']) * ones(T_pred), fillcolor=:red, alpha=0.35, label="constraints", legend=:topleft)
        end
        if y_max !== nothing
            plot!(Array(0:T_pred-1), y_max[i, :], fillrange=maximum([y_pred_max; y_test']) * ones(T_pred), fillcolor=:red, alpha=0.35, label="constraints")
        end

        title!("\$y_{$i}\$: predicted output vs. true output")
        ylabel!("\$y_{$i}\$")
        xlabel!("\$t\$")
        display(p)
    end
end

"""
    plot_autocorrelation(PMCMC_samples::Vector{PMCMC_sample}; max_lag=0)

Plot the autocorrelation function (ACF) of the PMCMC samples. This might be helpful when adjusting the thinning parameter ``k_d``.

# Arguments
- `PMCMC_samples`: PMCMC samples
- `max_lag`: maximum lag at which to calculate the ACF
"""
function plot_autocorrelation(PMCMC_samples::Vector{PMCMC_sample}; max_lag::Int=100)
    # Get number of models.
    K = size(PMCMC_samples, 1)

    # Get number of parameters of the PMCMC samples.
    n_variables = length(PMCMC_samples[1].theta) + size(PMCMC_samples[1].x_m1, 1)

    # Fill matrix with the series of the parameters of the PMCMC samples.
    sample_matrix = Array{Float64}(undef, K, n_variables)
    for i in 1:K
        # Sample state at the last timestep of the training dataset.
        star = sample(1:length(PMCMC_samples[i].w_m1), Weights(PMCMC_samples[i].w_m1))
        x_m1 = PMCMC_samples[i].x_m1[:, star]
        sample_matrix[i, :] .= [PMCMC_samples[i].theta; vec(x_m1)]
    end

    # Calculate the autocorrelation.
    autocorrelation = autocor(sample_matrix, Array(0:max_lag); demean=true)

    # Plot the ACF.
    p = plot(yticks=-1:0.1:1)
    for i in 1:n_variables
        if i == 1
            # Plot the ACF of the elements of theta.
            plot!(Array(0:max_lag), autocorrelation[:, i], lc=:red, lw=2, label="\$\\theta\$")
        elseif 1 < i <= length(PMCMC_samples[1].theta)
            plot!(Array(0:max_lag), autocorrelation[:, i], lc=:red, lw=2, label="")
        elseif i == length(PMCMC_samples[1].theta) + 1
            # Plot the ACF of the elements of x_t-1.
            plot!(Array(0:max_lag), autocorrelation[:, i], lc=:green, lw=2, label="\$x(t-1)\$")
        elseif length(PMCMC_samples[1].theta) + 1 < i
            plot!(Array(0:max_lag), autocorrelation[:, i], lc=:green, lw=2, label="")
        end
    end

    title!("Autocorrelation Function (AFC)")
    xlabel!("Lag")
    ylabel!("AFC")
    display(p)
end

"""
    plot_parameter_trace(PMCMC_samples::Vector{PMCMC_sample})
Plot the trace of the parameters of the PMCMC samples.
# Arguments
- `PMCMC_samples`: PMCMC samples
"""
function plot_parameter_trace(PMCMC_samples::Vector{PMCMC_sample})
    # Get number of models.
    K = size(PMCMC_samples, 1)

    # Get number of parameters of the PMCMC samples.
    n_variables = length(PMCMC_samples[1].theta) + size(PMCMC_samples[1].x_m1, 1)

    # Fill matrix with the series of the parameters of the PMCMC samples.
    sample_matrix = Array{Float64}(undef, K, n_variables)
    for i in 1:K
        # Sample state at the last timestep of the training dataset.
        star = sample(1:length(PMCMC_samples[i].w_m1), Weights(PMCMC_samples[i].w_m1))
        x_m1 = PMCMC_samples[i].x_m1[:, star]
        sample_matrix[i, :] .= [PMCMC_samples[i].theta; vec(x_m1)]
    end

    # Plot the trace of the parameters.
    for i in 1:n_variables
        p = plot(Array(0:K-1), sample_matrix[:, i], lw=2, legend=false)
        if i <= length(PMCMC_samples[1].theta)
            title!("Trace of \$\\theta_{$i}\$")
            ylabel!("\$\\theta_{$i}\$")
        else
            title!("Trace of \$x_{$(i-length(PMCMC_samples[1].theta))}\$")
            ylabel!("\$x_{$(i-length(PMCMC_samples[1].theta))}(t-1)\$")
        end
        xlabel!("Iteration")
        display(p)
    end
end

"""
    plot_parameter_pdf(PMCMC_samples::Vector{PMCMC_sample}; bins = 50, prior_pdf::Union{Nothing,Vector{Tuple{Vector{Float64},Vector{Float64}}}}=nothing, true_values=nothing)

Plots an histogram (empirical probability density function (PDF) estimate) for the parameters and the initial state (t=0). If provided, overlays the prior density for each variable and the true value.

# Arguments
- `PMCMC_samples`: PMCMC samples
- `bins`: number of bins to use for the histogram
- `prior_pdf`: vector of prior density values for each parameter and latent initial state
- `true_values`: true values for each parameter and latent initial state
"""
function plot_parameter_pdf(PMCMC_samples::Vector{PMCMC_sample}; bins::Int=50, prior_pdf::Union{Nothing,Vector{Tuple{Vector{Float64},Vector{Float64}}}}=nothing, true_values::AbstractVector{<:AbstractFloat}=nothing)
    # Get number of models.
    K = size(PMCMC_samples, 1)

    # Get number of parameters of the PMCMC samples.
    n_variables = length(PMCMC_samples[1].theta) + size(PMCMC_samples[1].x_m1, 1)

    # Fill matrix with the series of the parameters of the PMCMC samples.
    sample_matrix = Array{Float64}(undef, K, n_variables)
    for i in 1:K
        # Sample state at the last timestep of the training dataset.
        #=
        star = sample(1:length(PMCMC_samples[i].w_m1), Weights(PMCMC_samples[i].w_m1))
        x_m1 = PMCMC_samples[i].x_m1[:, star]
        sample_matrix[i, :] .= [PMCMC_samples[i].theta; vec(x_m1)]
        =#

        # Sample state at the first timestep of the training dataset.
        star = sample(1:length(PMCMC_samples[i].w_0), Weights(PMCMC_samples[i].w_0))
        x_0 = PMCMC_samples[i].x_0[:, star]
        sample_matrix[i, :] .= [PMCMC_samples[i].theta; vec(x_0)]
    end

    for i in 1:n_variables
        p = histogram(sample_matrix[:, i], bins=bins, normalize=:pdf, label="Posterior")

        # Plot prior if provided
        if !isnothing(prior_pdf) && i <= length(prior_pdf)
            value, density = prior_pdf[i]
            plot!(value, density, label="Prior", linestyle=:dash, lw=2)
        end

        # Plot true value if provided
        if !isnothing(true_values) && i <= length(true_values)
            plot!([true_values[i]], seriestype=:vline, label="True", lw=2)
        end

        if i <= length(PMCMC_samples[1].theta)
            title!("Sample PDF of \$\\theta_{$i}\$")
            xlabel!("\$\\theta_{$i}\$")
        else
            title!("Sample PDF of \$x_{$(i-length(PMCMC_samples[1].theta))}(t=0)\$")
            xlabel!("\$x_{$(i-length(PMCMC_samples[1].theta))}\$")
        end
        ylabel!("Density")
        display(p)
    end
end