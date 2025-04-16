"""
    particle_filter(u, y, n_x, N, f::Function, g::Function, sample_v::Function, log_pdf_w::Function, sample_x_init::Function)

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
- `sample_x_init`: function that returns a sample from the distribution over initial states; has no inputs

# Returns
- `x_pf`: state trajectories of particles
- `w`: normalized weights of particles
- `log_likelihood`: log-marginal likelihood estimate
"""
function particle_filter(u, y, n_x, N, f::Function, g::Function, sample_v::Function, log_pdf_w::Function, sample_x_init::Function)
    # Initialize and pre-allocate.
    T = size(y, 2)
    w = Array{Float64}(undef, T, N)
    x_pf = Array{Float64}(undef, n_x, N, T)
    a = Array{Int64}(undef, T, N)
    log_w = Array{Float64}(undef, 1, N)
    log_likelihood = 0.0

    # Sample initial states.
    for n in 1:N
        x_pf[:, n, 1] .= sample_x_init()
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
    function particle_MMH(u, y, n_x, K, K_b, k_d, N, f_theta::Function, g_theta::Function, sample_x_init::Function, sample_v_theta::Function, log_pdf_w_theta::Function, log_pdf_theta::Function, propose_theta::Function, log_ratio_proposal_pdf::Function, theta_init; print_progress=true)

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
- `sample_x_init`: function that returns a sample from the distribution over initial states; has no inputs
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
function particle_MMH(u, y, n_x, K, K_b, k_d, N, f_theta::Function, g_theta::Function, sample_x_init::Function, sample_v_theta::Function, log_pdf_w_theta::Function, log_pdf_theta::Function, propose_theta::Function, log_ratio_proposal_pdf::Function, theta_init; print_progress=true)
    # Total number of samples to be generated
    K_total = K_b + 1 + (K - 1) * (k_d + 1)

    # Get number of parameters, etc.
    n_theta = length(theta_init)
    n_u = size(u, 1)
    T = size(y, 2)

    # Initialize and pre-allocate.
    PMMH_samples = Vector{PMMH_sample}(undef, K)
    for k in 1:K
        PMMH_samples[k] = PMMH_sample(Array{Float64}(undef, n_theta), Array{Float64}(undef, n_x, N), Array{Float64}(undef, N), Array{Float64}(undef, n_u))
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
        x_pf, w, log_likelihood_prop = particle_filter(u, y, n_x, N, f, g, sample_v, log_pdf_w, sample_x_init)

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
    staged_PMMH(u, y, n_x, K, K_b, k_d, N, f_theta::Function, g_theta::Function, sample_x_init::Function, sample_v_theta::Function, log_pdf_w_theta::Function, log_pdf_theta::Function, theta_init, proposal_cov_init, T_chunk, K_stage, alpha; print_progress=true, regularizer=1e-8)

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
- `sample_x_init`: function that returns a sample from the distribution over initial states; has no inputs
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
function staged_PMMH(u, y, n_x, K, K_b, k_d, N, f_theta::Function, g_theta::Function, sample_x_init::Function, sample_v_theta::Function, log_pdf_w_theta::Function, log_pdf_theta::Function, theta_init, proposal_cov_init, T_chunk, K_stage, alpha; print_progress=true, regularizer=1e-8)
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
            PMMH_samples_stage, acceptance_ratio_stage = particle_MMH(u_i, y_i, n_x, K_stage, K_b, 0, N, f_theta, g_theta, sample_x_init, sample_v_theta, log_pdf_w_theta, log_pdf_theta, propose_theta, log_ratio_proposal_pdf, theta; print_progress=false)[1:2]
        else
            # In the final stage, sample K samples with thinning parameter k_d.
            PMMH_samples_stage, acceptance_ratio_stage = particle_MMH(u_i, y_i, n_x, K, K_b, k_d, N, f_theta, g_theta, sample_x_init, sample_v_theta, log_pdf_w_theta, log_pdf_theta, propose_theta, log_ratio_proposal_pdf, theta; print_progress=false)[1:2]
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
    test_prediction(PMMH_samples::Vector{PMMH_sample}, n_x, f_theta::Function, g_theta::Function, sample_v_theta::Function, sample_w_theta::Function, u_test, y_test)

Simulate the PMMH samples forward in time and compare the predictions to the test data.

# Arguments
- `PMMH_samples`: PMMH samples
- `n_x`: number of states
- `f_theta`: state transition function parametrized by theta; has inputs (theta, x, u)
- `g_theta`: measurement function parametrized by theta; has inputs (theta, x, u)
- `sample_v_theta`: function that returns N samples from the process noise distribution parametrized by theta; has input (theta, N)
- `sample_w_theta`: function that returns N samples from the measurement noise distribution parametrized by theta; has input (theta, N)
- `k_n`: each model is simulated ``k_n`` times
- `u_test`: test input
- `y_test`: test output
"""
function test_prediction(PMMH_samples::Vector{PMMH_sample}, n_x, f_theta::Function, g_theta::Function, sample_v_theta::Function, sample_w_theta::Function, k_n, u_test, y_test)
    println("### Testing model")

    # Get number of models, etc.
    K = size(PMMH_samples, 1)
    n_y = size(y_test, 1)
    T_test = size(y_test, 2)

    # Pre-allocate.
    x_test_sim = Array{Float64}(undef, n_x, T_test + 1, K, k_n)
    y_test_sim = Array{Float64}(undef, n_y, T_test, K, k_n)

    # Simulate models forward.
    Threads.@threads for k in 1:K
        # Get current model.
        f(x, u) = f_theta(PMMH_samples[k].theta, x, u)
        g(x, u) = g_theta(PMMH_samples[k].theta, x, u)
        sample_v(N) = sample_v_theta(PMMH_samples[k].theta, N)
        sample_w(N) = sample_w_theta(PMMH_samples[k].theta, N)

        # Simulate each model k_n times.
        for kn in 1:k_n
            # Pre-allocate.
            x_loop = Array{Float64}(undef, n_x, T_test + 1)
            y_loop = Array{Float64}(undef, n_y, T_test)

            # Sample initial state.
            star = sample(1:length(PMMH_samples[k].w_m1), Weights(PMMH_samples[k].w_m1))
            x_m1 = PMMH_samples[k].x_m1[:, star]
            x_loop[:, 1] .= f(x_m1, PMMH_samples[k].u_m1) + sample_v(1)

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
    @printf("Mean rmse: %.2f\n", mean_rmse)
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
function plot_predictions(y_pred, y_test; plot_percentiles=false, y_min=nothing, y_max=nothing)
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
            plot!(Array(0:T_pred-1), y_min', fillrange=minimum([y_pred_min; y_test']) * ones(T_pred), fillcolor=:red, alpha=0.35, label="constraints", legend=:topleft)
        end
        if y_max !== nothing
            plot!(Array(0:T_pred-1), y_max', fillrange=maximum([y_pred_max; y_test']) * ones(T_pred), fillcolor=:red, alpha=0.35, label="constraints")
        end

        # Add title, labels...
        if 1 < n_y
            title!("y_" * string(i) * ": predicted output vs. true output")
            ylabel!("y_" * string(i))
        else
            title!("predicted output vs. true output")
            ylabel!("y")
        end
        xlabel!("t")
        display(p)
    end
end

"""
    plot_autocorrelation(PMMH_samples::Vector{PMMH_sample}; max_lag=0)

Plot the autocorrelation function (ACF) of the PMMH samples. This might be helpful when adjusting the thinning parameter ``k_d``.

# Arguments
- `PMMH_samples`: PMMH samples
- `max_lag`: maximum lag at which to calculate the ACF
"""
function plot_autocorrelation(PMMH_samples::Vector{PMMH_sample}; max_lag=0)
    # Get number of models.
    K = size(PMMH_samples, 1)

    # Get number of parameters of the PG samples.
    number_of_variables = length(PMMH_samples[1].theta) + size(PMMH_samples[1].x_m1, 1)

    if max_lag == 0
        max_lag = K - 1
    end

    # Fill matrix with the series of the parameters of the PG samples.
    signal_matrix = Array{Float64}(undef, K, number_of_variables)
    for i in 1:K
        # Sample initial state.
        star = sample(1:length(PMMH_samples[i].w_m1), Weights(PMMH_samples[i].w_m1))
        x_m1 = PMMH_samples[i].x_m1[:, star]
        signal_matrix[i, :] .= [PMMH_samples[i].theta; vec(x_m1)]
    end

    # Calculate the autocorrelation.
    autocorrelation = autocor(signal_matrix, Array(0:max_lag); demean=true)

    # Plot the ACF.
    p = plot(yticks=-1:0.1:1)
    for i in 1:number_of_variables

        if i == 1
            # Plot the ACF of the elements of theta.
            plot!(Array(0:max_lag), autocorrelation[:, i], lc=:red, lw=2, label="theta")
        elseif 1 < i <= length(PMMH_samples[1].theta)
            plot!(Array(0:max_lag), autocorrelation[:, i], lc=:red, lw=2, label="")
        elseif i == length(PMMH_samples[1].theta) + 1
            # Plot the ACF of the elements of x_t-1.
            plot!(Array(0:max_lag), autocorrelation[:, i], lc=:green, lw=2, label="x")
        elseif length(PMMH_samples[1].theta) + 1 < i
            plot!(Array(0:max_lag), autocorrelation[:, i], lc=:green, lw=2, label="")
        end
    end

    title!("Autocorrelation Function (AFC)")
    xlabel!("Lag")
    ylabel!("AFC")
    display(p)
end