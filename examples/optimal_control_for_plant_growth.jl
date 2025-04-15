using LinearAlgebra
using Random
using Distributions
using Plots

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
K = 50 # number of PMMH samples per stage
k_d = 5 # number of samples to be skipped to decrease correlation (thinning)
K_b = 200 # length of burn-in period for each stage
N = 50 # number of particles of the particle filter
T_chunk = 5 # number of datapoints added at each stage
K_stage = 500 # number of samples per stage
alpha = 0.01 # scaling of the proposal covariance
regularizer = 0 # regularizer for proposal covariance

# Number of states, etc.
n_x = 3 # number of states
n_u = 2 # number of control inputs
n_y = 2 # number of outputs

# State transition function.
f_theta(theta, x, u) = SIMPLE.f_theta(theta, x, u)

# Zero-mean Gaussian process noise with variance Q - assumed to be known (without loss of generality).
Q = Diagonal([0.01, 5, 5]) # variance of process noise
sample_v_theta(theta, N) = rand(MvNormal(zeros(n_x), Q), N) # sample process noise

# Measurement function - assumed to be known (without loss of generality).
g_theta(theta, x, u) = [1 0 0; 0 1 0] * x # observation function

# Zero-mean Gaussian measurement noise with known variance R - normalizing factors are ommited as they cancel out in the acceptance ratio.
R = Diagonal([0.1^2, 13]) # variance of zero-mean Gaussian measurement noise
sample_w_theta(theta, N) = rand(MvNormal(zeros(n_y), R), N) # sample measurement noise
log_pdf_w_theta(theta, w) = -0.5 * sum(w .* (R \ w), dims=1) # log pdf of measurement noise, scaling 

# Prior for parameters.
theta_mean = [
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

theta_var = [
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
x_init_mean = [0, 0, 50] # mean
x_init_var = Diagonal([1e-6, 1e-6, 1]) # variance
sample_x_init() = rand(MvNormal(x_init_mean, x_init_var))

# Parameters for data generation.
T_train = 50 # number of days for training
T_test = 50  # number of days used for testing (via forward simulation - see below)
T_total = T_train + T_test

# Generate training data.
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

x[:, 1] = sample_x_init() # random initial state
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

PMMH_samples = PMMHopt.staged_PMMH(u_training, y_training, n_x, K, K_b, k_d, N, f_theta, g_theta, sample_x_init, sample_v_theta, log_pdf_w_theta, log_pdf_theta, theta_init, theta_cov, T_chunk, K_stage, alpha; regularizer=regularizer)