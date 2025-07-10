"""
    adapt_N(PMCMC_samples::Vector{PMCMC_sample}, u::AbstractMatrix{<:AbstractFloat}, y::AbstractMatrix{<:AbstractFloat}, n_x::Int, N::Int, f_theta::Function, g_theta::Function, sample_x_0::Function, sample_v_theta::Function, log_pdf_w_theta::Function; K_adapt::Int=1, num_runs::Int=100, target_var::AbstractFloat=2.0)

Estimates the variance of the log-likelihood from repeated runs of the particle filter and returns a recommended new particle number N based on a target variance level.

# Arguments
- `PMCMC_samples`: PMCMC samples
- `u`: training input trajectory
- `y`: training output trajectory
- `n_x`: number of states
- `K`: number of models/scenarios to be sampled
- `K_b`: length of the burn in period
- `k_d`: number of models/scenarios to be skipped to decrease correlation (thinning)
- `N`: number of particles
- `f_theta`: state transition function parametrized by theta; has inputs (theta, x, u)
- `g_theta`: measurement function parametrized by theta; has inputs (theta, x, u)
- `sample_x_0`: function that returns a sample from the distribution over initial states; has no inputs
- `sample_v_theta`: function that returns N samples from the process noise distribution parametrized by theta; has input (theta, N)
- `log_pdf_w_theta`: function that returns the logarithm of the probability density function of the measurement noise parametrized by theta; has inputs (theta, w)
- `K_adapt`: number of posterior samples used for the adaptation of the number of particles
- `num_runs`: number of PF runs for variance estimation (default: 100)
- `target_var`: target variance for log-likelihood (default: 2.0)

# Returns
- `N_suggested`: recommended number of particles
- 'log_likelihood_var_avg`: average variance of the log-likelihood
"""
function adapt_N(PMCMC_samples::Vector{PMCMC_sample}, u::AbstractMatrix{<:AbstractFloat}, y::AbstractMatrix{<:AbstractFloat}, n_x::Int, N::Int, f_theta::Function, g_theta::Function, sample_x_0::Function, sample_v_theta::Function, log_pdf_w_theta::Function; K_adapt::Int=1, num_runs::Int=100, target_var::AbstractFloat=2.0)
    K = size(PMCMC_samples, 1)
    if K_adapt > K
        warning("K_adapt is larger than the provided number of samples K. Using K instead.")
        K_adapt = K
    end

    indices = shuffle(1:K)[1:K_adapt]
    log_likelihood_vars = Float64[]

    for k in indices
        # Update model.
        f(x, u) = f_theta(PMCMC_samples[k].theta, x, u)
        g(x, u) = g_theta(PMCMC_samples[k].theta, x, u)
        sample_v(N) = sample_v_theta(PMCMC_samples[k].theta, N)
        log_pdf_w(w) = log_pdf_w_theta(PMCMC_samples[k].theta, w)

        # Compute variance of log-likelihood.
        log_likelihoods = zeros(num_runs)
        for i in 1:num_runs
            _, _, log_likelihood = particle_filter(u, y, n_x, N, f, g, sample_v, log_pdf_w, sample_x_0)
            log_likelihoods[i] = log_likelihood
        end
        push!(log_likelihood_vars, var(log_likelihoods, corrected=true))
    end
    log_likelihood_var_avg = mean(log_likelihood_vars)
    N_suggested = max(1, ceil(Int, N * log_likelihood_var_avg / target_var))
    return N_suggested, log_likelihood_var_avg
end

"""
    adapt_N_blocked(PMCMC_samples::Vector{PMCMC_sample}, u::AbstractMatrix{<:AbstractFloat}, y::AbstractMatrix{<:AbstractFloat}, n_x::Int, N::Int, f_theta::Function, g_theta::Function, sample_v_theta::Function, log_pdf_w_theta::Function; K_adapt::Int=1, num_runs::Int=100, target_var::AbstractFloat=2.0)

Estimates the variance of the log-likelihood from repeated runs of the particle filter and returns a recommended new particle number N based on a target variance level.
This function is similar to `adapt_N`, but it is designed for the blocked PMCMC sampler, where the initial state is fixed and needs to be passed to this function.

# Arguments
- `PMCMC_samples`: PMCMC samples
- `u`: training input trajectory
- `y`: training output trajectory
- `n_x`: number of states
- `K`: number of models/scenarios to be sampled
- `K_b`: length of the burn in period
- `k_d`: number of models/scenarios to be skipped to decrease correlation (thinning)
- `N`: number of particles
- `f_theta`: state transition function parametrized by theta; has inputs (theta, x, u)
- `g_theta`: measurement function parametrized by theta; has inputs (theta, x, u)
- `sample_v_theta`: function that returns N samples from the process noise distribution parametrized by theta; has input (theta, N)
- `log_pdf_w_theta`: function that returns the logarithm of the probability density function of the measurement noise parametrized by theta; has inputs (theta, w)
- `K_adapt`: number of posterior samples used for the adaptation of the number of particles
- `num_runs`: number of PF runs for variance estimation (default: 100)
- `target_var`: target variance for log-likelihood (default: 2.0)

# Returns
- `N_suggested`: recommended number of particles
- 'log_likelihood_var_avg`: average variance of the log-likelihood
"""
function adapt_N_blocked(PMCMC_samples::Vector{PMCMC_sample}, u::AbstractMatrix{<:AbstractFloat}, y::AbstractMatrix{<:AbstractFloat}, n_x::Int, N::Int, f_theta::Function, g_theta::Function, sample_v_theta::Function, log_pdf_w_theta::Function; K_adapt::Int=1, num_runs::Int=100, target_var::AbstractFloat=2.0)
    K = size(PMCMC_samples, 1)
    if K_adapt > K
        warning("K_adapt is larger than the provided number of samples K. Using K instead.")
        K_adapt = K
    end

    indices = shuffle(1:K)[1:K_adapt]
    log_likelihood_vars = Float64[]

    for k in indices
        # Update model.
        f(x, u) = f_theta(PMCMC_samples[k].theta, x, u)
        g(x, u) = g_theta(PMCMC_samples[k].theta, x, u)
        sample_v(N) = sample_v_theta(PMCMC_samples[k].theta, N)
        log_pdf_w(w) = log_pdf_w_theta(PMCMC_samples[k].theta, w)
        sample_x_0() = PMCMC_samples[k].x_0

        # Compute variance of log-likelihood.
        log_likelihoods = zeros(num_runs)
        for i in 1:num_runs
            _, _, log_likelihood = particle_filter(u, y, n_x, N, f, g, sample_v, log_pdf_w, sample_x_0)
            log_likelihoods[i] = log_likelihood
        end
        push!(log_likelihood_vars, var(log_likelihoods, corrected=true))
    end
    log_likelihood_var_avg = mean(log_likelihood_vars)
    N_suggested = max(1, ceil(Int, N * log_likelihood_var_avg / target_var))
    return N_suggested, log_likelihood_var_avg
end