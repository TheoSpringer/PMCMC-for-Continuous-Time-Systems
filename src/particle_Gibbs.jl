"""
    particle_filter(u, y, n_x, N, f::Function, g::Function, sample_v::Function, log_pdf_w::Function, sample_x_init::Function)

Run a particle filter with ancestor sampling to approximate the log-marginal likelihood ``\\log p(y_{0:t} \\mid \\theta, \\{u_{0:t}\\})``.

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
    log_w = Array{Float64}(undef, N)
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
        w[t, :] .= exp.(log_w .- max_log_w)

        # Estimate log-likelihood.
        log_likelihood += log(sum(w[t, :])) + max_log_w - log(N)

        # Normalize weights.
        w[t, :] .= w[t, :] ./ sum(w[t, :])
    end
    return x_pf, w, log_likelihood
end


"""
    particle_MMH(u, y, K, K_b, k_d, N, f_theta::Function, p_theta::Function, propose_theta::Function, theta_init, sample_x_init::Function, sample_process_noise, g::Function, R; x_prim=nothing)

Run particle marginal Metropolis-Hastings (PMMH) with ancestor sampling to obtain samples ``\\{\\theta, x_{T:-1}\\}^{[1:K]}`` from the joint parameter and state posterior distribution ``p(\\theta, x_{T:-1} \\mid \\mathbb{D}=\\{u_{T:-1}, y_{T:-1}\\})``.

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
- `pdf_v_theta`: probability density function of the process noise parametrized by theta; has inputs (theta, v)
- `log_pdf_w_theta`: function that returns the logarithm of the probability density function of the measurement noise parametrized by theta; has inputs (theta, w)
- `p_theta`: probability density function of theta (prior); has input (theta)
- `propose_theta`: function that proposes new theta (proposal distribution); has input (theta)
- `theta_init`: initial theta
- `sample_x_init`: function that returns a sample from the distribution over initial states; has no inputs
- `x_prim`: prespecified trajectory for the first iteration

This function is based on the paper

    Andrieu, Christophe, Arnaud Doucet, and Roman Holenstein. "Particle Markov chain Monte Carlo methods." Journal of the Royal Statistical Society Series B: Statistical Methodology 72.3 (2010): 269-342.
"""
function particle_MMH(u, y, n_x, K, K_b, k_d, N, f_theta::Function, g_theta::Function, sample_v_theta::Function, pdf_v_theta::Function, log_pdf_w_theta::Function, p_theta::Function, propose_theta::Function, theta_init, sample_x_init::Function)
    # Total number of models
    K_total = K_b + 1 + (K - 1) * (k_d + 1)

    # Get number of inputs, etc.
    n_u = size(u, 1)
    T = size(y, 2)

    # Initialize and pre-allocate.
    PMMH_samples = Vector{PG_sample}(undef, K)
    for k in 1:K
        PMMH_samples[k] = PG_sample(Array{Float64}(undef, size(theta_init)), Array{Float64}(undef, N), Array{Float64}(undef, n_x, N), Array{Float64}(undef, n_u))
    end
    accepted_samples = 0
    theta = theta_init

    # Time PMMH sampler.
    learning_timer = time()

    println("### Started PMMH sampling")

    while accepted_samples < K_total
        # Propose new parameters.
        theta_prop = propose_theta(theta)
        if !(p_theta(theta_prop) > 0)
            continue
        end

        # Update model, i.e., update state transition and observation function and noise distributions.
        f(x, u) = f_theta(theta_prop, x, u)
        g(x, u) = g_theta(theta_prop, x, u)
        sample_v(N) = sample_v_theta(theta, N)
        pdf_v(v) = pdf_v_theta(theta, v)
        log_pdf_w(w) = log_pdf_w_theta(theta, w)

        # Run particle smoother.
        x_pf, w, a = particle_smoother(u, y, n_x, N, f, g, sample_v, log_pdf_w, x_prim, pdf_v)


        # Sample state trajectory x_T:-1 to condition on.
        star = sample(1:N, Weights(w[end, :]))
        x_prim[:, T] .= x_pf[:, star, T]
        for t in T-1:-1:1
            star = a[t+1, star]
            x_prim[:, t] .= x_pf[:, star, t]
        end

        # Use sample if the burn-in period is reached and the sample is not removed by thinning.
        if K_b < k && (mod(k - (K_b + 1), k_d + 1) == 0)
            PG_samples[current_sample].A .= A
            PG_samples[current_sample].Q .= Q
            PG_samples[current_sample].w_m1 .= w[end, :]
            PG_samples[current_sample].x_m1 .= x_pf[:, :, end]
            PG_samples[current_sample].u_m1 .= u[:, end]
            current_sample += 1
        end

        # Sample new model parameters conditional on sampled trajectory, i.e., sample from p(A, Q | x_T:-1).
        zeta .= x_prim[:, 2:T]
        z .= phi(x_prim[:, 1:T-1], u[:, 1:T-1])
        Phi .= zeta * zeta' # statistic; see paper "A flexible state-space model for learning nonlinear dynamical systems"
        Psi .= zeta * z' # statistic
        Sigma .= z * z' # statistic
        A, Q = MNIW_sample(Phi, Psi, Sigma, V, Lambda_Q, ell_Q, T - 1) # sample new model parameters

        # Print progress.
        @printf("Accepted sample %i/%i\n", accepted_samples, K_total)
    end



    # Print runtime.
    time_learning = time() - learning_timer
    @printf("### Learning complete\nRuntime: %.2f s\n", time_learning)

    return PG_samples
end

"""
    test_prediction(PG_samples::Vector{PG_sample}, phi::Function, g, R, k_n, u_test, y_test)

Simulate the PGS samples forward in time and compare the predictions to the test data.

# Arguments
- `PG_samples`: PG samples
- `phi`: basis functions
- `g`: observation function
- `R`: variance of zero-mean Gaussian measurement noise
- `k_n`: each model is simulated ``k_n`` times
- `u_test`: test input
- `y_test`: test output
"""
function test_prediction(PG_samples::Vector{PG_sample}, phi::Function, g, R, k_n, u_test, y_test)
    println("### Testing model")

    # Get number of models, etc.
    K = size(PG_samples, 1)
    n_x = size(PG_samples[1].A, 1)
    n_y = size(y_test, 1)
    T_test = size(y_test, 2)

    # Measurement noise distribution
    mvn_e = MvNormal(zeros(n_y), R)

    # Pre-allocate.
    x_test_sim = Array{Float64}(undef, n_x, T_test + 1, K, k_n)
    y_test_sim = Array{Float64}(undef, n_y, T_test, K, k_n)

    # Simulate models forward.
    Threads.@threads for k in 1:K
        # Get current model.
        A = PG_samples[k].A
        Q = PG_samples[k].Q
        f(x, u) = A * phi(x, u)
        mvn_v = MvNormal(zeros(n_x), Q) # process noise distribution

        # Simulate each model k_n times.
        for kn in 1:k_n
            # Pre-allocate.
            x_loop = Array{Float64}(undef, n_x, T_test + 1)
            y_loop = Array{Float64}(undef, n_y, T_test)

            # Sample initial state.
            star = sample(1:length(PG_samples[k].w_m1), Weights(PG_samples[k].w_m1))
            x_m1 = PG_samples[k].x_m1[:, star]
            x_loop[:, 1] .= f(x_m1, PG_samples[k].u_m1) + rand(mvn_v)

            # Simulate model forward.
            for t in 1:T_test
                if t >= 2
                    x_loop[:, t] .= f(x_loop[:, t-1], u_test[:, t-1]) + rand(mvn_v)
                end
                y_loop[:, t] .= g(x_loop[:, t], u_test[:, t]) + rand(mvn_e)
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
    mean_rmse = sqrt(mean((y_test_sim .- repeat(y_test', 1, 1, K * k_n)) .^ 2))
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
    plot_autocorrelation(PG_samples::Vector{PG_sample}; max_lag=0)

Plot the autocorrelation function (ACF) of the PG samples. This might be helpful when adjusting the thinning parameter ``k_d``.

# Arguments
- `PG_samples`: PG samples
- `max_lag`: maximum lag at which to calculate the ACF
"""
function plot_autocorrelation(PG_samples::Vector{PG_sample}; max_lag=0)
    # Get number of models.
    K = size(PG_samples, 1)

    # Get number of parameters of the PG samples.
    number_of_variables = length(PG_samples[1].A) + length(PG_samples[1].Q) + size(PG_samples[1].x_m1, 1)

    if max_lag == 0
        max_lag = K - 1
    end

    # Fill matrix with the series of the parameters of the PG samples.
    signal_matrix = Array{Float64}(undef, K, number_of_variables)
    for i in 1:K
        # Sample initial state.
        star = sample(1:length(PG_samples[i].w_m1), Weights(PG_samples[i].w_m1))
        x_m1 = PG_samples[i].x_m1[:, star]
        signal_matrix[i, :] .= [vec(PG_samples[i].A); vec(PG_samples[i].Q); vec(x_m1)]
    end

    # Calculate the autocorrelation.
    autocorrelation = autocor(signal_matrix, Array(0:max_lag); demean=true)

    # Plot the ACF.
    p = plot(yticks=-1:0.1:1)
    for i in 1:number_of_variables
        # Plot the ACF of the elements of A.
        if i == 1
            plot!(Array(0:max_lag), autocorrelation[:, i], lc=:red, lw=2, label="A")
        elseif 1 < i <= length(PG_samples[1].A)
            plot!(Array(0:max_lag), autocorrelation[:, i], lc=:red, lw=2, label="")
            # Plot the ACF of the elements of Q.  
        elseif i == length(PG_samples[1].A) + 1
            plot!(Array(0:max_lag), autocorrelation[:, i], lc=:blue, lw=2, label="Q")
        elseif (length(PG_samples[1].A) + 1 < i) && (i <= length(PG_samples[1].A) + length(PG_samples[1].Q))
            plot!(Array(0:max_lag), autocorrelation[:, i], lc=:blue, lw=2, label="")
            # Plot the ACF of the elements of x_t-1.
        elseif i == length(PG_samples[1].A) + length(PG_samples[1].Q) + 1
            plot!(Array(0:max_lag), autocorrelation[:, i], lc=:green, lw=2, label="x")
        elseif length(PG_samples[1].A) + length(PG_samples[1].Q) + 1 < i
            plot!(Array(0:max_lag), autocorrelation[:, i], lc=:green, lw=2, label="")
        end
    end

    title!("Autocorrelation Function (AFC)")
    xlabel!("Lag")
    ylabel!("AFC")
    display(p)
end