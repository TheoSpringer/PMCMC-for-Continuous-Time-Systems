using LinearAlgebra
using Random
using Distributions
using Plots
using StatsPlots
using Printf

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
N = 50 # number of particles of the particle filter
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

#=
theta_mean = [
    2150,   # tau_sum
    485,    # Ia
    313,    # Ib
    6.0,    # theta_base
    27.0,   # theta_opt
    1.14 * 1e-3,    # RUE
    100.0,  # Iheat
    6.0,    # Iwater
    33.0,   # theta_heat
    46.0,   # theta_ext
    0.06,   # Sco2
    1.9,    # Swater
    0.95,   # Rmax
]

theta_var = [
    205000,   # tau_sum
    4725,    # Ia
    2969,    # Ib
    1.188,    # theta_base
    0.250,   # theta_opt
    0.108 * 1e-3,    # RUE
    1e-6,  # Iheat
    4.686,    # Iwater
    0.750,   # theta_heat
    4.688,   # theta_ext
    6.750 * 1e-4,   # Sco2
    0.743,    # Swater
    0.100,   # Rmax
]
=#

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

#=
theta_true = [
    2800,   # tau_sum
    520,    # Ia
    400,    # Ib
    6.0,    # theta_base
    26.0,   # theta_opt
    1.00 * 1e-3,    # RUE
    100.0,  # Iheat
    5.0,    # Iwater
    32.0,   # theta_heat
    45.0,   # theta_ext
    0.07,   # Sco2
    2.5,    # Swater
    0.95,   # Rmax
]
=#

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

# Run a staged PMMH sampler. The acceptance ratio should be around 25 % for a random walk proposal.
PMMH_samples, acceptance_ratio, time_sampling = PMMHopt.staged_PMMH(u_training, y_training, n_x, K, K_b, k_d, N, f_theta, g_theta, sample_x_0, sample_v_theta, log_pdf_w_theta, log_pdf_theta, theta_init, theta_cov, T_chunk, K_stage, alpha; regularizer=regularizer)

# Test the models with the test data by simulating it forward in time. The predictions should fit the test data well.
PMMHopt.test_prediction(PMMH_samples, n_x, f_theta, g_theta, sample_v_theta, sample_w_theta, 1, u_test, y_test)

# Plot autocorrelation of the PMMH samples. The ACF should ideally decay quickly. With right thinning, the autocorrelation should be close to 0 also for small lags.
PMMHopt.plot_autocorrelation(PMMH_samples; max_lag=200)

# Print effective sample size (ESS) of the PMMH samples. With right thinning, ESS should be close to K.
ess = PMMHopt.compute_ess(PMMH_samples; max_lag=200)
@printf("Minimum ESS: %.1f\n", minimum(ess))

# Plot parameter trace. After the burn in is removed the trace should not have any trends and should be stationary. Also, the step size should seem resonable.
PMMHopt.plot_parameter_trace(PMMH_samples)

# Plot histogram and the priors. If the data is informative, the posterior should contract with respect to the prior to the true value.
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