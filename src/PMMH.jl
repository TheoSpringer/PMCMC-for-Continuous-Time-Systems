"""
    particle_filter(u, y, n_x, N, f::Function, g::Function, sample_v::Function, log_pdf_w::Function, sample_x_0::Function)

Run a particle filter to approximate the log-marginal likelihood ``\\log p(y_{0:t_0-1} \\mid \\theta, \\{u_{0:t_0-1}\\})``.

# Arguments
- `u`: training input trajectory
- `y`: training output trajectory
- `n_x`: number of states
- `N`: number of particles
- `f`: state transition function; has inputs (x, u)
- `g`: measurement function; has inputs (x, u)
- `sample_v`: function that returns N samples from the process noise distribution; has input (N)
- `pdf_v`: probability density function of the process noise; has input (v)
- `log_pdf_w`: function that returns the logarithm of the probability density function of the measurement noise; has input (w)
- `sample_x_0`: function that returns a sample from the distribution over initial states; has no inputs

# Returns
- `x_pf`: state trajectories of particles
- `w`: normalized weights of particles
- `log_likelihood`: log-marginal likelihood estimate
"""
function particle_filter(u, y, n_x, N, f::Function, g::Function, sample_v::Function, log_pdf_w::Function, sample_x_0::Function)
    # Initialize and pre-allocate.
    T = size(y, 2)
    w = Array{Float64}(undef, T, N)
    x_pf = Array{Float64}(undef, n_x, N, T)
    a = Array{Int64}(undef, T, N)
    log_w = Array{Float64}(undef, 1, N)
    log_likelihood = 0.0

    # Sample initial states.
    for n in 1:N
        x_pf[:, n, 1] .= sample_x_0()
    end

    # Particle filter.
    for t in 1:T
        if t >= 2
            # Resample particles.
            a[t, :] .= sample(1:N, Weights(w[t-1, :]), N)

            # Propagate resampled particles.
            x_pf[:, :, t] .= f(x_pf[:, a[t, :], t-1], repeat(u[:, t-1], 1, N)) + sample_v(N)
        end

        # PF weight update based on measurement model (logarithms are used for numerical reasons).
        log_w .= log_pdf_w(y[:, t] .- g(x_pf[:, :, t], repeat(u[:, t], 1, N)))
        max_log_w = maximum(log_w)
        w[[t], :] .= exp.(log_w .- max_log_w)
        sum_w = sum(w[t, :])
        w[t, :] .= w[t, :] ./ sum_w

        # Estimate log-likelihood.
        log_likelihood += log(sum_w) + max_log_w - log(N)
    end
    return x_pf, w, log_likelihood
end


"""
    function particle_MMH(u, y, n_x, K, K_b, k_d, N, f_theta::Function, g_theta::Function, sample_x_0::Function, sample_v_theta::Function, log_pdf_w_theta::Function, log_pdf_theta::Function, propose_theta::Function, log_ratio_proposal_pdf::Function, theta_init; print_progress=true)

Run particle marginal Metropolis-Hastings (PMMH) to obtain samples ``\\{\\theta, x_{0:t_0-1}\\}^{[1:K]}`` from the joint parameter and state posterior distribution ``p(\\theta, x_{0:t_0-1} \\mid \\mathbb{D}=\\{u_{0:t_0-1}, y_{0:t_0-1}\\})``.

# Arguments
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
- `log_pdf_theta`: function that returns the logarithm of the probability density function of theta (prior); has input (theta)
- `propose_theta`: function that proposes new theta (proposal distribution); has input (theta)
- `log_ratio_proposal_pdf`: function that returns the logarithm of the ratio of proposal densities; has input arguments (theta_accepted, theta_prop)
- `theta_init`: initial theta
- `print_progress`: if set to true, the progress is printed

# Returns
- `PMMH_samples`: PMMH samples
- `time_sampling`: sampling time
- `acceptance_ratio`: acceptance ratio of the PMMH sampler

## References
- Andrieu, Christophe, Arnaud Doucet, and Roman Holenstein. "Particle Markov chain Monte Carlo methods." Journal of the Royal Statistical Society Series B: Statistical Methodology 72.3 (2010): 269-342.
"""
function particle_MMH(u, y, n_x, K, K_b, k_d, N, f_theta::Function, g_theta::Function, sample_x_0::Function, sample_v_theta::Function, log_pdf_w_theta::Function, log_pdf_theta::Function, propose_theta::Function, log_ratio_proposal_pdf::Function, theta_init; print_progress=true)
    # Total number of samples to be generated
    K_total = K_b + 1 + (K - 1) * (k_d + 1)

    # Get number of parameters, etc.
    n_theta = length(theta_init)
    n_u = size(u, 1)
    T = size(y, 2)

    # Initialize and pre-allocate.
    PMMH_samples = Vector{PMMH_sample}(undef, K)
    for k in 1:K
        PMMH_samples[k] = PMMH_sample(Array{Float64}(undef, n_theta), Array{Float64}(undef, n_x, N), Array{Float64}(undef, N), Array{Float64}(undef, n_u), Array{Float64}(undef, n_x, N), Array{Float64}(undef, N))
    end
    accepted_samples = 0
    current_sample = 1
    n_proposals = 0
    theta = theta_init
    log_likelihood = -Inf
    log_p_theta = -Inf

    # Time PMMH sampler.
    sampling_timer = time()

    if print_progress
        println("### Started PMMH sampling")
    end

    while accepted_samples < K_total
        # Propose new parameters.
        n_proposals += 1
        theta_prop = propose_theta(theta)
        log_p_theta_prop = log_pdf_theta(theta_prop)[]
        if !isfinite(log_p_theta_prop)
            continue
        end

        # Update model, i.e., update state transition and observation function and noise distributions.
        f(x, u) = f_theta(theta_prop, x, u)
        g(x, u) = g_theta(theta_prop, x, u)
        sample_v(N) = sample_v_theta(theta_prop, N)
        log_pdf_w(w) = log_pdf_w_theta(theta_prop, w)

        # Run particle filter.
        x_pf, w, log_likelihood_prop = particle_filter(u, y, n_x, N, f, g, sample_v, log_pdf_w, sample_x_0)

        # Compute acceptance probability.
        log_acceptance_ratio = log_likelihood_prop - log_likelihood + log_p_theta_prop - log_p_theta + log_ratio_proposal_pdf(theta, theta_prop)

        # Accept or reject the proposal.
        if log(rand()) < log_acceptance_ratio
            accepted_samples += 1
            theta = theta_prop
            log_p_theta = log_p_theta_prop
            log_likelihood = log_likelihood_prop

            # Use sample if the burn-in period is reached and the sample is not removed by thinning.
            if (K_b < accepted_samples) && (mod(accepted_samples - (K_b + 1), k_d + 1) == 0)
                PMMH_samples[current_sample].theta .= theta_prop
                PMMH_samples[current_sample].x_m1 .= x_pf[:, :, end]
                PMMH_samples[current_sample].w_m1 .= w[end, :]
                PMMH_samples[current_sample].u_m1 .= u[:, end]
                PMMH_samples[current_sample].x_0 .= x_pf[:, :, 1]
                PMMH_samples[current_sample].w_0 .= w[1, :]
                current_sample += 1
            end

            # Print progress.
            if print_progress
                @printf("\e[32m%i/%i samples accepted\e[0m\n", accepted_samples, K_total)
            end
        else
            # Print progress.
            if print_progress
                @printf("\e[31m%i/%i samples accepted\e[0m\n", accepted_samples, K_total)
            end
        end
    end

    time_sampling = time() - sampling_timer
    acceptance_ratio = ((K_total - 1) / n_proposals) * 100

    # Print results.
    if print_progress
        @printf("### PMMH sampling complete\nRuntime: %.2f s\nAcceptance ratio: %.2f %%\n", time_sampling, acceptance_ratio)
    end

    return PMMH_samples, acceptance_ratio, time_sampling
end

"""
    staged_PMMH(u, y, n_x, K, K_b, k_d, N, f_theta::Function, g_theta::Function, sample_x_0::Function, sample_v_theta::Function, log_pdf_w_theta::Function, log_pdf_theta::Function, theta_init, proposal_cov_init, T_chunk, K_stage, alpha; print_progress=true, regularizer=1e-8)

Run particle marginal Metropolis-Hastings (PMMH) with incremental data and adaptive proposal to obtain samples ``\\{\\theta, x_{0:t_0-1}\\}^{[1:K]}`` from the joint parameter and state posterior distribution ``p(\\theta, x_{0:t_0-1} \\mid \\mathbb{D}=\\{u_{0:t_0-1}, y_{0:t_0-1}\\})``.
The number of data points used in the likelihood computation is gradually increased by a fixed chunk size. At each stage, the MMH sampler is run on the current data subset, and the proposal distribution is adapted based on the empirical covariance of the collected samples.

# Arguments
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
- `log_pdf_theta`: function that returns the logarithm of the probability density function of theta (prior); has input (theta)
- `theta_init`: initial theta
- `proposal_cov_init`: initial covariance (matrix) for the multivariate normal proposal
- `T_chunk`: number of data points added at each stage
- `K_stage`: number of samples per stage
- `alpha`: proposal scaling factor (scalar or vector; if a vector its i‐th element is used at stage i)
- `print_progress`: if set to true, the progress is printed
- `regularizer`: small constant added to the diagonal of the proposal covariance

# Returns
- `PMMH_samples`: final samples from full-data posterior
- `acceptance_ratio`: vector containing the acceptance ratio of each stage
"""
function staged_PMMH(u, y, n_x, K, K_b, k_d, N, f_theta::Function, g_theta::Function, sample_x_0::Function, sample_v_theta::Function, log_pdf_w_theta::Function, log_pdf_theta::Function, theta_init, proposal_cov_init, T_chunk, K_stage, alpha; print_progress=true, regularizer=1e-8)
    # Get number of parameters, etc.
    n_theta = length(theta_init)
    T = size(y, 2)

    # Determine the number of stages (each stage adds T_chunk data points)
    N_stages = ceil(Int, T / T_chunk)

    # Initialize current parameter vector
    theta = theta_init
    proposal_cov = proposal_cov_init

    # Allocate an array to store samples from each stage.
    acceptance_ratio = zeros(N_stages)
    PMMH_samples = Vector{PMMH_sample}(undef, K)

    sampling_timer = time()

    if print_progress
        println("### Started staged PMMH sampling")
    end

    for i in 1:N_stages
        # Select current data chunk - use data from 1 to T_i.
        T_i = min(i * T_chunk, T)
        u_i = u[:, 1:T_i]
        y_i = y[:, 1:T_i]

        # Define the proposal function as sampling from a multivariate normal.
        propose_theta(theta) = rand(MvNormal(theta, proposal_cov))
        log_ratio_proposal_pdf(theta_accepted, theta_prop) = 0

        # Call the base PMMH sampler.
        if i < N_stages
            # For intermediate stages, sample K_stage samples without thinning.
            PMMH_samples_stage, acceptance_ratio_stage = particle_MMH(u_i, y_i, n_x, K_stage, K_b, 0, N, f_theta, g_theta, sample_x_0, sample_v_theta, log_pdf_w_theta, log_pdf_theta, propose_theta, log_ratio_proposal_pdf, theta; print_progress=false)[1:2]
        else
            # In the final stage, sample K samples with thinning parameter k_d.
            PMMH_samples_stage, acceptance_ratio_stage = particle_MMH(u_i, y_i, n_x, K, K_b, k_d, N, f_theta, g_theta, sample_x_0, sample_v_theta, log_pdf_w_theta, log_pdf_theta, propose_theta, log_ratio_proposal_pdf, theta; print_progress=false)[1:2]
        end

        # Save stage samples and acceptance ratio.
        acceptance_ratio[i] = acceptance_ratio_stage

        if i < N_stages
            # Update proposal covariance based on the empirical covariance
            Theta = zeros(n_theta, K_stage)
            for j in 1:K_stage
                Theta[:, j] = PMMH_samples_stage[j].theta
            end

            post_cov_theta = cov(transpose(Theta))
            if isa(alpha, Number)
                proposal_cov = alpha * post_cov_theta + regularizer * Matrix(I, n_theta, n_theta)
            else
                proposal_cov = alpha[i] * post_cov_theta + regularizer * Matrix(I, n_theta, n_theta)
            end

            # Update the current state to the last sample from the current stage.
            theta = Theta[:, end]
        else
            PMMH_samples = PMMH_samples_stage
        end
        if print_progress
            @printf("Stage %i/%i complete\nAcceptance ratio: %.2f %%\n", i, N_stages, acceptance_ratio_stage)
        end
    end

    average_acceptance_ratio = mean(acceptance_ratio)
    time_sampling = time() - sampling_timer
    if print_progress
        @printf("### Staged PMMH sampling complete\nRuntime: %.2f s\nAverage acceptance ratio: %.2f %%\n",
            time_sampling, average_acceptance_ratio)
    end
    return PMMH_samples, acceptance_ratio, time_sampling
end

"""
    compute_ess(PMMH_samples::Vector{PMMH_sample}; max_lag=100)

Compute the effective sample size (ESS) for each parameter and state.

# Arguments
- `PMMH_samples`: PMMH samples
- `max_lag`: maximum lag for autocorrelation estimation.

# Returns
- `ess`: vector of ESS estimates for all variables.
"""
function compute_ess(PMMH_samples::Vector{PMMH_sample}; max_lag=100)
    # Get number of models.
    K = size(PMMH_samples, 1)

    # Get number of parameters of the PMMH samples.
    number_of_variables = length(PMMH_samples[1].theta) + size(PMMH_samples[1].x_m1, 1)

    # Fill matrix with the series of the parameters of the PMMH samples.
    sample_matrix = Array{Float64}(undef, K, number_of_variables)
    for i in 1:K
        # Sample initial state.
        star = sample(1:length(PMMH_samples[i].w_m1), Weights(PMMH_samples[i].w_m1))
        x_m1 = PMMH_samples[i].x_m1[:, star]
        sample_matrix[i, :] .= [PMMH_samples[i].theta; vec(x_m1)]
    end

    # Calculate the autocorrelation.
    autocorrelation = autocor(sample_matrix, Array(0:max_lag); demean=true)
    ess = zeros(number_of_variables)

    for i in 1:number_of_variables
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
    compute_gelman_rubin(PMMH_chains::Vector{Vector{PMMH_sample}})

Compute the Gelman–Rubin statistic for each parameter and latent state from a vector of PMMH chains.

# Arguments
- `PMMH_chains`: vector of chains, where each chain is a vector of PMMH samples

# Returns
- `R_hat`: vector of R̂ values, one for each variable
"""
function compute_gelman_rubin(PMMH_chains::Vector{Vector{PMMH_sample}})
    M = length(PMMH_chains) # Number of chains
    K = length(PMMH_chains[1]) # Number of samples per chain

    # Get number of parameters of the PMMH samples.
    number_of_variables = length(PMMH_chains[1][1].theta) + size(PMMH_chains[1][1].x_m1, 1)

    # Extract samples from each chain
    sample_matrices_chains = Array{Float64}[]
    for chain in PMMH_chains
        sample_matrix = Array{Float64}(undef, K, number_of_variables)

        for i in 1:K
            # Sample state at the last timestep of the training dataset.
            star = sample(1:length(chain[i].w_m1), Weights(chain[i].w_m1))
            x_m1 = chain[i].x_m1[:, star]
            sample_matrix[i, :] .= [chain[i].theta; vec(x_m1)]
        end

        push!(sample_matrices_chains, sample_matrix)
    end

    # Compute means and variances
    means = zeros(M, number_of_variables)
    variances = zeros(M, number_of_variables)
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