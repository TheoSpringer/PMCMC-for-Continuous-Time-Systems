using LinearAlgebra
using Random
using Distributions
using Plots
using StatsPlots
using Printf
using Base.Threads

include("SIMPLE/SIMPLE.jl")
include("TOMGRO/TOMGRO.jl")
include("../src/PMMHopt.jl")

using .SIMPLE
using .TOMGRO
using .PMMHopt

# Specify seed (for reproducible results).
Random.seed!(1)

# Time PMMH algorithm.
sampling_timer = time()

# Learning parameters.
K = Int(1e4) # number of PMMH samples in final stage
k_d = 0 # number of samples to be skipped to decrease correlation (thinning)
K_b = 200 # length of burn-in period for each stage
N_init = 200 # initial number of particles of the particle filter - will be adjusted later
T_chunk = 2 # number of datapoints added at each stage
K_stage = Int(1e3) # number of samples per stage
alpha = collect(range(3, stop=0.1, length=20)) # scaling of the proposal covariance
regularizer = 0 # regularizer for proposal covariance

# Number of states, etc.
n_x = 3 # number of states
n_u = 2 # number of control inputs
n_y = 2 # number of outputs

# State transition function.
f_theta(theta, x, u) = SIMPLE.f_theta(theta, x, u)

# Zero-mean Gaussian process noise with variance Q - assumed to be known (without loss of generality).
Q = Diagonal([0.01^2, 0.1^2, 0.1^2]) # variance of process noise
sample_v_theta(theta, N) = rand(MvNormal(zeros(n_x), Q), N) # sample process noise

# Measurement function - assumed to be known (without loss of generality).
g_theta(theta, x, u) = [1 0 0; 0 1 0] * x # observation function

# Zero-mean Gaussian measurement noise with known variance R - normalizing factors are ommited as they cancel out in the acceptance ratio.
R = Diagonal([0.1^2, 1^2]) # variance of zero-mean Gaussian measurement noise
sample_w_theta(theta, N) = rand(MvNormal(zeros(n_y), R), N) # sample measurement noise
log_pdf_w_theta(theta, w) = -0.5 * sum(w .* (R \ w), dims=1) # log pdf of measurement noise, scaling 

# Prior for parameters.
theta_mean = [
    2550,   # tau_sum
    535,    # Ia
    350,    # Ib
]

theta_var = [
    62500,   # tau_sum
    225,    # Ia
    2500,    # Ib
]

# Log pdf of prior - normalizing factors are ommited as they cancel out in the acceptance ratio.
theta_cov = Diagonal(theta_var) # covariance matrix of prior
log_pdf_theta(theta) = -0.5 * sum((theta - theta_mean) .* (theta_cov \ (theta - theta_mean)), dims=1)

# Initial proposal distribution.
log_ratio_proposal_pdf(theta_accepted, theta_prop) = 0
proposal_variance_scaling = 1e-3 # scaling factor for the proposal variance
propose_theta(theta) = rand(MvNormal(theta, proposal_variance_scaling * theta_cov))

# Initial guess for model parameters.
theta_init = theta_mean

# Normally distributed initial state
x_0_mean = [0, 0, 50] # mean
x_0_var = [1e-6, 1e-6, 1] # variance
sample_x_0() = rand(MvNormal(x_0_mean, Diagonal(x_0_var)))
log_pdf_x_0(x_0) = -0.5 * sum((x_0 - x_0_mean) .* (Diagonal(x_0_var) \ (x_0 - x_0_mean)), dims=1)

# Initial guess for initial state. Only relevant for blocked PMMH.
x_0_init = x_0_mean

# Parameters for data generation.
T_train = 40 # number of days for training
T_test = 40  # number of days used for testing (via forward simulation - see below)
T_total = T_train + T_test

# Generate training data.
theta_true = [
    2800,   # tau_sum
    520,    # Ia
    400,    # Ib
]

f_true(x, u) = f_theta(theta_true, x, u) # true state transition function
g_true(x, u) = g_theta(theta_true, x, u) # true measurement function
R_true = R # true measurement noise variance
sample_v_true(N) = sample_v_theta(theta_true, N)
sample_w_true(N) = sample_w_theta(theta_true, N)

# Input trajectory used to generate training and test data
u = [fill(25.0, T_total)'; fill(0.0, T_total)'; fill(25.0, T_total)']

# Generate data by forward simulation.
x = Array{Float64}(undef, n_x, T_total) # true latent state trajectory
y = Array{Float64}(undef, n_y, T_total) # output trajectory (measured)

x[:, 1] = sample_x_0() # random initial state
for t in 2:T_total
    x[:, t] = f_true(x[:, t-1], u[:, t-1]) + sample_v_true(1)
end

for t in 1:T_total
    y[:, t] = g_true(x[:, t], u[:, t]) + sample_w_true(1)
end

# Split data into training and test data.
u_training = u[:, 1:T_train]
x_training = x[:, 1:T_train]
y_training = y[:, 1:T_train]

u_test = u[:, T_train+1:end]
x_test = x[:, T_train+1:end]
y_test = y[:, T_train+1:end]

# Plot data.
plot()
for i in 1:n_u
    plot!(1:T_total, u[i, :], label="u_$i", lw=2, legend=:topright)
end
for i in 1:n_y
    plot!(1:T_total, y[i, :], label="y_$i", lw=2)
end
xlabel!("t")
ylabel!("u | y")

# Adjust number of particles for the particle filter. This requires a good estimate of the parameters theta.
#=
N_suggested = adapt_N(u, y, n_x, N_init, theta_true, f_theta, g_theta, sample_x_0, sample_v_theta, log_pdf_w_theta; num_runs=100, target_var=2)[1]
@printf("Suggested N: %i\n", N_suggested)
=#

# Run a staged PMMH sampler.
# Aim for an acceptance ratio of around 20–30% for a random-walk proposal.
PMMH_samples, acceptance_ratio, time_sampling = PMMHopt.staged_PMMH(u_training, y_training, n_x, K, K_b, k_d, N_init, f_theta, g_theta, sample_x_0, sample_v_theta, log_pdf_w_theta, log_pdf_theta, theta_init, theta_cov, T_chunk, K_stage, alpha; regularizer=regularizer, K_adapt=10)

# In case the uncertainty about the initial state is large and/or the process noise is small, it may be beneficial to use a blocked PMMH sampler.
# proposal_cov_init = Diagonal(vcat(theta_var, x_0_var)) # initial proposal covariance for theta and x_0
# PMMH_samples, acceptance_ratio, time_sampling = PMMHopt.staged_PMMH_blocked(u_training, y_training, n_x, K, K_b, k_d, N_init, f_theta, g_theta, sample_v_theta, log_pdf_w_theta, log_pdf_theta, log_pdf_x_0, theta_init, x_0_init, proposal_cov_init, T_chunk, K_stage, alpha; regularizer=regularizer, K_adapt=10)

# Simulate the posterior models forward and compare to test data.
# The predicted trajectories should track the true outputs well.
PMMHopt.test_prediction(PMMH_samples, n_x, f_theta, g_theta, sample_v_theta, sample_w_theta, 1, u_test, y_test)

# Plot the autocorrelation function (ACF) of the samples.
# A well-mixed chain will show fast decay of autocorrelation. After thinning, the ACF should be near zero even at small lags.
PMMHopt.plot_autocorrelation(PMMH_samples; max_lag=200)

# Compute the effective sample size (ESS).
# The ESS indicates how many effectively independent samples were drawn. Ideally, after thinning, ESS should approach K.
# The goal of tuning is to maximize the ESS per second.
ess = PMMHopt.compute_ess(PMMH_samples; max_lag=200)
@printf("Minimum ESS: %.1f (= %.2f / s)\n", minimum(ess), minimum(ess) / time_sampling)

# Plot the parameter and latent state trace.
# The trace should appear stationary and show no long-term trends after burn-in. Jump sizes should look reasonable.
PMMHopt.plot_parameter_trace(PMMH_samples)

# Plot the posterior histogram with overlaid priors and true values (if known).
# If the data is informative, the posterior should be tighter than the prior and centered near the true value.
prior_pdf = Vector{Tuple{Vector{Float64},Vector{Float64}}}()
for i in 1:length(theta_mean)
    prior = Normal(theta_mean[i], sqrt(theta_var[i]))
    values = range(quantile(prior, 0.01), stop=quantile(prior, 0.99), length=500)
    push!(prior_pdf, (values, pdf(prior, values)))
end
for i in 1:length(x_0_mean)
    prior = Normal(x_0_mean[i], sqrt(x_0_var[i]))
    values = range(quantile(prior, 0.01), stop=quantile(prior, 0.99), length=500)
    push!(prior_pdf, (values, pdf(prior, values)))
end
PMMHopt.plot_parameter_pdf(PMMH_samples; bins=50, prior_pdf=prior_pdf, true_values=[theta_true; x_training[:, 1]])

# Optional: run multiple independent PMMH chains and compute the Gelman–Rubin statistic.
# R̂ quantifies convergence by comparing within-chain to between-chain variance.
# R̂ close to 1 (typically R̂ < 1.05) indicates good convergence across chains.
#=
M = 10 # number of independent chains
PMMH_chains = Vector{Vector{PMMH_sample}}(undef, M)
@threads for m in 1:M
    theta_init = rand(MvNormal(theta_mean, Diagonal(theta_var)))
    PMMH_chains[m] = PMMHopt.staged_PMMH(u_training, y_training, n_x, K, K_b, k_d, N, f_theta, g_theta, sample_x_0, sample_v_theta, log_pdf_w_theta, log_pdf_theta, theta_init, theta_cov, T_chunk, K_stage, alpha; regularizer=regularizer)[1]
end

R_hat = PMMHopt.compute_gelman_rubin(PMMH_chains)
@printf("Maximum R̂: %.2f\n", maximum(R_hat))
=#

