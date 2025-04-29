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

    # Get number of parameters of the PMMH samples.
    number_of_variables = length(PMMH_samples[1].theta) + size(PMMH_samples[1].x_m1, 1)

    if max_lag == 0
        max_lag = K - 1
    end

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

"""
    plot_parameter_trace(PMMH_samples::Vector{PMMH_sample})
Plot the trace of the parameters of the PMMH samples.
# Arguments
- `PMMH_samples`: PMMH samples
"""
function plot_parameter_trace(PMMH_samples::Vector{PMMH_sample})
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

    # Plot the trace of the parameters.
    for i in 1:number_of_variables
        plot(Array(0:K-1), sample_matrix[:, i])
        if i <= length(PMMH_samples[1].theta)
            title!("Trace of theta_$i")
            ylabel!("theta_$i")
        else
            title!("Trace of x_$(i-length(PMMH_samples[1].theta))")
            ylabel!("x_$(i-length(PMMH_samples[1].theta))")
        end
        xlabel!("Iteration")
        display(p)
    end
end

"""
    plot_parameter_pdf(PMMH_samples::Vector{PMMH_sample}; bins = 50)

Plots an histrogram for the parameters (empirical probability density fuction (PDF) estimate).

# Arguments
- `PMMH_samples`: PMMH samples
- `bins`: number of bins to use for the histogram
"""
function plot_parameter_pdf(PMMH_samples::Vector{PMMH_sample}; bins=50)
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

    # Plot the trace of the parameters.
    for i in 1:number_of_variables
        plot(Array(0:K-1), sample_matrix[:, i])
        if i <= length(PMMH_samples[1].theta)
            title!("Trace of theta_$i")
            ylabel!("theta_$i")
        else
            title!("Trace of x_$(i-length(PMMH_samples[1].theta))")
            ylabel!("x_$(i-length(PMMH_samples[1].theta))")
        end
        xlabel!("Iteration")
        display(p)
    end

    for i in 1:number_of_variables
        histogram(sample_matrix[:, i], bins=bins, normalize=:pdf)
        if i <= length(PMMH_samples[1].theta)
            title!("Sample PDF of theta_$i")
            xlabel!("theta_$i")
        else
            title!("Sample PDF of x_$(i-length(PMMH_samples[1].theta))")
            xlabel!("x_$(i-length(PMMH_samples[1].theta))")
        end
        ylabel!("Sample PDF")
        display(p)
    end
end
