"""
    particle_MMH_blocked(u::AbstractMatrix{<:AbstractFloat}, y::AbstractMatrix{<:AbstractFloat}, n_x::Int, K::Int, K_b::Int, k_d::Int, N::Int, f_theta::Function, g_theta::Function, sample_v_theta::Function, log_pdf_w_theta::Function, log_pdf_theta::Function, log_pdf_x_0::Function, propose_theta_x_0::Function, log_ratio_proposal_pdf::Function, theta_init::AbstractVector{<:AbstractFloat}, x_0_init::AbstractVector{<:AbstractFloat}; print_progress::Bool=true)

Run blocked particle marginal Metropolis-Hastings (PMMH) to obtain samples ``\\{\\theta, x_{0:t_0-1}\\}^{[1:K]}`` from the joint parameter and state posterior distribution ``p(\\theta, x_{0:t_0-1} \\mid \\mathbb{D}=\\{u_{0:t_0-1}, y_{0:t_0-1}\\})``.
The difference between this function and `particle_MMH` is that the initial state is part of the proposal distribution. This can be beneficial if there is large uncertainty about the initial state and little process noise.

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
- `sample_v_theta`: function that returns N samples from the process noise distribution parametrized by theta; has input (theta, N)
- `log_pdf_w_theta`: function that returns the logarithm of the probability density function of the measurement noise parametrized by theta; has inputs (theta, w)
- `log_pdf_theta`: function that returns the logarithm of the probability density function of theta (prior); has input (theta)
- `log_pdf_x_0`: function that returns the logarithm of the probability density function of the initial state (prior); has input (x_0)
- `propose_theta_x_0`: function that proposes new theta and x_0 (proposal distribution); has input (theta, x_0)
- `log_ratio_proposal_pdf`: function that returns the logarithm of the ratio of proposal densities; has input arguments (theta_accepted, x_0_accepted, theta_prop, x_0_prop)
- `theta_init`: initial theta
- `x_0_init`: initial x_0
- `print_progress`: if set to true, the progress is printed

# Returns
- `PMMH_samples`: PMMH samples
- `time_sampling`: sampling time
- `acceptance_ratio`: acceptance ratio of the PMMH sampler
"""
function particle_MMH_blocked(u::AbstractMatrix{<:AbstractFloat}, y::AbstractMatrix{<:AbstractFloat}, n_x::Int, K::Int, K_b::Int, k_d::Int, N::Int, f_theta::Function, g_theta::Function, sample_v_theta::Function, log_pdf_w_theta::Function, log_pdf_theta::Function, log_pdf_x_0::Function, propose_theta_x_0::Function, log_ratio_proposal_pdf::Function, theta_init::AbstractVector{<:AbstractFloat}, x_0_init::AbstractVector{<:AbstractFloat}; print_progress::Bool=true)
    # Total number of samples to be generated
    K_total = K_b + 1 + (K - 1) * (k_d + 1)

    # Get number of parameters, etc.
    n_theta = length(theta_init)
    n_u = size(u, 1)
    T = size(y, 2)

    # Initialize and pre-allocate.
    PMMH_samples = Vector{PMCMC_sample}(undef, K)
    for k in 1:K
        PMMH_samples[k] = PMCMC_sample(Array{Float64}(undef, n_theta), Array{Float64}(undef, n_x, N), Array{Float64}(undef, N), Array{Float64}(undef, n_u), Array{Float64}(undef, n_x, 1), Array{Float64}(undef, 1))
    end
    accepted_samples = 0
    current_sample = 1
    n_proposals = 0
    theta = theta_init
    x_0 = x_0_init
    log_likelihood = -Inf
    log_p_theta = -Inf
    log_p_x_0 = -Inf

    # Time PMMH sampler.
    sampling_timer = time()

    if print_progress
        println("### Started blocked PMMH sampling")
    end

    while accepted_samples < K_total
        # Propose new parameters and initial state.
        n_proposals += 1
        theta_x_0_prop = propose_theta_x_0(theta, x_0)
        theta_prop = theta_x_0_prop[1:n_theta]
        x_0_prop = theta_x_0_prop[n_theta+1:end]
        log_p_theta_prop = log_pdf_theta(theta_prop)[]
        log_p_x_0_prop = log_pdf_x_0(x_0_prop)[]
        if !isfinite(log_p_theta_prop + log_p_x_0_prop)
            continue
        end

        # Update model, i.e., update state transition and observation function and noise distributions.
        f(x, u) = f_theta(theta_prop, x, u)
        g(x, u) = g_theta(theta_prop, x, u)
        sample_v(N) = sample_v_theta(theta_prop, N)
        log_pdf_w(w) = log_pdf_w_theta(theta_prop, w)
        sample_x_0() = x_0_prop

        # Run particle filter.
        x_pf, w, log_likelihood_prop = particle_filter(u, y, n_x, N, f, g, sample_v, log_pdf_w, sample_x_0)

        # Compute acceptance probability.
        log_acceptance_ratio = log_likelihood_prop - log_likelihood + log_p_theta_prop - log_p_theta + log_p_x_0_prop - log_p_x_0 + log_ratio_proposal_pdf(theta, x_0, theta_prop, x_0_prop)

        # Accept or reject the proposal.
        if log(rand()) < log_acceptance_ratio
            accepted_samples += 1
            theta = theta_prop
            x_0 = x_0_prop
            log_likelihood = log_likelihood_prop
            log_p_theta = log_p_theta_prop
            log_p_x_0 = log_p_x_0_prop

            # Use sample if the burn-in period is reached and the sample is not removed by thinning.
            if (K_b < accepted_samples) && (mod(accepted_samples - (K_b + 1), k_d + 1) == 0)
                @views begin
                    PMMH_samples[current_sample].theta .= theta_prop
                    PMMH_samples[current_sample].x_m1 .= x_pf[:, end, :]
                    PMMH_samples[current_sample].w_m1 .= w[end, :]
                    PMMH_samples[current_sample].u_m1 .= u[:, end]
                    PMMH_samples[current_sample].x_0 .= x_0
                    PMMH_samples[current_sample].w_0 .= [1.0]
                end
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

    # Print runtime and acceptance ratio.
    if print_progress
        @printf("### Blocked PMMH sampling complete\nRuntime: %.2f s\nAcceptance ratio: %.2f %%\n", time_sampling, acceptance_ratio)
    end

    return PMMH_samples, acceptance_ratio, time_sampling
end

"""
    staged_PMMH_blocked(u::AbstractMatrix{<:AbstractFloat}, y::AbstractMatrix{<:AbstractFloat}, n_x::Int, K::Int, K_b::Int, k_d::Int, N::Int, f_theta::Function, g_theta::Function, sample_v_theta::Function, log_pdf_w_theta::Function, log_pdf_theta::Function, log_pdf_x_0::Function, theta_init::AbstractVector{<:AbstractFloat}, x_0_init::AbstractVector{<:AbstractFloat}, proposal_cov_init::AbstractMatrix{<:AbstractFloat}, T_chunk::Int, K_stage::Int, alpha::Union{AbstractFloat,AbstractVector{<:AbstractFloat}}; print_progress::Bool=true, regularizer::AbstractFloat=1e-8, K_adapt::Int=0, num_runs_N_adapt::Int=100, target_var::AbstractFloat=2.0, min_N::Int=20)

Run blocked particle marginal Metropolis-Hastings (PMMH) with incremental data and adaptive proposal to obtain samples ``\\{\\theta, x_{0:t_0-1}\\}^{[1:K]}`` from the joint parameter and state posterior distribution ``p(\\theta, x_{0:t_0-1} \\mid \\mathbb{D}=\\{u_{0:t_0-1}, y_{0:t_0-1}\\})``.
The number of data points used in the likelihood computation is gradually increased by a fixed chunk size. At each stage, the MMH sampler is run on the current data subset, and the proposal distribution is adapted based on the empirical covariance of the collected samples.
The difference between this function and `staged_PMMH` is that the initial state is part of the proposal distribution. This can be beneficial if there is large uncertainty about the initial state and little process noise.

# Arguments
- `u`: training input trajectory
- `y`: training output trajectory
- `n_x`: number of states
- `K`: number of models/scenarios to be sampled
- `K_b`: length of the burn in period
- `k_d`: number of models/scenarios to be skipped to decrease correlation (thinning)
- `N`: (initial) number of particles; might be adjusted based on the variance of the log-likelihood (see below)
- `f_theta`: state transition function parametrized by theta; has inputs (theta, x, u)
- `g_theta`: measurement function parametrized by theta; has inputs (theta, x, u)
- `sample_v_theta`: function that returns N samples from the process noise distribution parametrized by theta; has input (theta, N)
- `log_pdf_w_theta`: function that returns the logarithm of the probability density function of the measurement noise parametrized by theta; has inputs (theta, w)
- `log_pdf_theta`: function that returns the logarithm of the probability density function of theta (prior); has input (theta)
- `log_pdf_x_0`: function that returns the logarithm of the probability density function of the initial state (prior); has input (x_0)
- `theta_init`: initial theta
- `x_0_init`: initial x_0
- `proposal_cov_init`: initial covariance (matrix) for the multivariate normal proposal
- `T_chunk`: number of data points added at each stage
- `K_stage`: number of samples per stage
- `alpha`: proposal scaling factor (scalar or vector; if a vector its i‐th element is used at stage i)
- `print_progress`: if set to true, the progress is printed
- `regularizer`: small constant added to the diagonal of the proposal covariance
- `K_adapt`: number of posterior samples used for the adaptation of the number of particles; adaptation is deactivated if set to 0 (default: 0)
- `num_runs_N_adapt`: number of PF runs for the adaptation of the number of particles (default: 100)
- `target_var`: target variance for log-likelihood (default: 2.0)
- `min_N`: minimum number of particles (default: 20)

# Returns
- `PMMH_samples`: final samples from full-data posterior
- `acceptance_ratio`: vector containing the acceptance ratio of each stage
"""
function staged_PMMH_blocked(u::AbstractMatrix{<:AbstractFloat}, y::AbstractMatrix{<:AbstractFloat}, n_x::Int, K::Int, K_b::Int, k_d::Int, N::Int, f_theta::Function, g_theta::Function, sample_v_theta::Function, log_pdf_w_theta::Function, log_pdf_theta::Function, log_pdf_x_0::Function, theta_init::AbstractVector{<:AbstractFloat}, x_0_init::AbstractVector{<:AbstractFloat}, proposal_cov_init::AbstractMatrix{<:AbstractFloat}, T_chunk::Int, K_stage::Int, alpha::Union{AbstractFloat,AbstractVector{<:AbstractFloat}}; print_progress::Bool=true, regularizer::AbstractFloat=1e-8, K_adapt::Int=0, num_runs_N_adapt::Int=100, target_var::AbstractFloat=2.0, min_N::Int=20)
    # Get number of parameters, etc.
    n_theta = length(theta_init)
    T = size(y, 2)
    n_variables = length(theta_init) + n_x

    # Determine the number of stages (each stage adds T_chunk data points)
    N_stages = ceil(Int, T / T_chunk)

    # Initialize current parameter vector
    theta = theta_init
    x_0 = x_0_init
    proposal_cov = proposal_cov_init

    # Allocate an array to store samples from each stage.
    acceptance_ratio = zeros(N_stages)
    PMMH_samples = Vector{PMCMC_sample}(undef, K)

    # Store initial N in case of adaptation.
    if K_adapt > 0
        N_init = N
    end

    sampling_timer = time()

    if print_progress
        println("### Started staged blocked PMMH sampling")
    end

    for i in 1:N_stages
        # Select current data chunk - use data from 1 to T_i.
        T_i = min(i * T_chunk, T)
        u_i = u[:, 1:T_i]
        y_i = y[:, 1:T_i]

        # Define the proposal function as sampling from a multivariate normal.
        propose_theta_x_0(theta, x_0) = rand(MvNormal(vcat(theta, x_0), proposal_cov))
        log_ratio_proposal_pdf(theta, x_0, theta_prop, x_0_prop) = 0

        # Call the base PMMH sampler.
        if i < N_stages
            # For intermediate stages, sample K_stage samples without thinning.
            PMMH_samples_stage, acceptance_ratio_stage = particle_MMH_blocked(u_i, y_i, n_x, K_stage, K_b, k_d, N, f_theta, g_theta, sample_v_theta, log_pdf_w_theta, log_pdf_theta, log_pdf_x_0, propose_theta_x_0, log_ratio_proposal_pdf, theta, x_0; print_progress=false)[1:2]
        else
            # In the final stage, sample K samples with thinning parameter k_d.
            PMMH_samples_stage, acceptance_ratio_stage = particle_MMH_blocked(u_i, y_i, n_x, K, K_b, k_d, N, f_theta, g_theta, sample_v_theta, log_pdf_w_theta, log_pdf_theta, log_pdf_x_0, propose_theta_x_0, log_ratio_proposal_pdf, theta, x_0; print_progress=false)[1:2]
        end

        # Save stage samples and acceptance ratio.
        acceptance_ratio[i] = acceptance_ratio_stage

        if i < N_stages
            # Update proposal covariance based on the empirical covariance
            Theta_X_0 = hcat([vcat(s.theta, s.x_0) for s in PMMH_samples_stage]...)

            post_cov_theta = cov(transpose(Theta_X_0))
            if isa(alpha, Number)
                proposal_cov = alpha * post_cov_theta + regularizer * Matrix(I, n_variables, n_variables)
            else
                proposal_cov = alpha[i] * post_cov_theta + regularizer * Matrix(I, n_variables, n_variables)
            end

            # Update the current state to the last sample from the current stage.
            theta = Theta_X_0[1:n_theta, end]
            x_0 = Theta_X_0[n_theta+1:end, end]

            # Adapt the number of particles for the next stage.
            if K_adapt > 0
                N_suggested, log_likelihood_var_avg = adapt_N_blocked(PMMH_samples_stage, u_i, y_i, n_x, N_init, f_theta, g_theta, sample_v_theta, log_pdf_w_theta; K_adapt, num_runs=num_runs_N_adapt, target_var=target_var)
                N = max(min_N, N_suggested)
                if print_progress
                    @printf("Adjusted N to %i (avg. log-likelihood variance ≈ %.2f)\n", N, log_likelihood_var_avg)
                end
            end
        else
            PMMH_samples = PMMH_samples_stage
        end
        if print_progress
            @printf("Stage %i/%i complete\nAcceptance ratio: %.2f %%\n", i, N_stages, acceptance_ratio_stage)
        end
    end

    # Print runtime and average acceptance ratio.
    average_acceptance_ratio = mean(acceptance_ratio)
    time_sampling = time() - sampling_timer
    if print_progress
        @printf("### Staged PMMH sampling complete\nRuntime: %.2f s\nAverage acceptance ratio: %.2f %%\n",
            time_sampling, average_acceptance_ratio)
    end

    return PMMH_samples, acceptance_ratio, time_sampling
end