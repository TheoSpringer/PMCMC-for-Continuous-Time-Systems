using Test
using Revise
using LinearAlgebra
using Random
using Distributions
using Plots
using StatsPlots
using Printf
using JuMP
import HSL_jll

using PMMHopt

# Specify seed (for reproducible results).
Random.seed!(1)

# Time PMMH algorithm.
sampling_timer = time()

# Learning parameters.
K = Int(1e1) # Int(1e4) # number of PMMH samples in final stage
k_d = 0 # number of samples to be skipped to decrease correlation (thinning)
K_b = 200 # length of burn-in period for each stage
N_init = 200 # initial number of particles of the particle filter - will be adjusted later
T_chunk = 2 # number of datapoints added at each stage
K_stage = Int(1e3) # number of samples per stage
alpha = collect(range(5, stop=0.1, length=20)) # scaling of the proposal covariance
regularizer = 0.0 # regularizer for proposal covariance

# Number of states, etc.
n_x = 3 # number of states
n_u = 3 # number of control inputs
n_y = 2 # number of outputs

# State transition function.
f_theta(theta, x, u) = SIMPLE.f_theta(theta, x, u)

# Zero-mean Gaussian process noise with variance Q - assumed to be known (without loss of generality).
Q = Diagonal([0.01^2, 0.1^2, 0.1^2]) # variance of process noise
sample_v_theta(theta, N) = rand(MvNormal(zeros(n_x), Q), N) # sample process noise

# Measurement function - assumed to be known (without loss of generality).
const C = [1.0 0 0; 0 1 0]
g_theta(theta, x, u) = C * x # observation function

# Zero-mean Gaussian measurement noise with known variance R - normalizing factors are ommited as they cancel out in the acceptance ratio.
R = Diagonal([0.1^2, 1^2]) # variance of zero-mean Gaussian measurement noise
sample_w_theta(theta, N) = rand(MvNormal(zeros(n_y), R), N) # sample measurement noise
log_pdf_w_theta(theta, w) = -0.5 * sum(w .* (R \ w), dims=1) # log pdf of measurement noise, scaling 

# Prior for parameters.
theta_mean = [
    2550.0,   # tau_sum
    535.0    # Ia
]

theta_var = [
    62500.0,   # tau_sum
    225.0    # Ia
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
x_0_mean = [0.0, 0.0, 350.0] # mean
x_0_var = [1e-3, 1e-3, 12500.0] # variance
sample_x_0() = rand(MvNormal(x_0_mean, Diagonal(x_0_var)))
log_pdf_x_0(x_0) = -0.5 * sum((x_0 - x_0_mean) .* (Diagonal(x_0_var) \ (x_0 - x_0_mean)), dims=1)

# Initial guess for initial state. Only relevant for blocked PMMH.
# x_0_init = x_0_mean

# Parameters for data generation.
T_train = 40 # number of days for training
T_test = 40  # number of days used for testing (via forward simulation - see below)
T_total = T_train + T_test

# Generate training data.
theta_true = [
    2800.0,   # tau_sum
    520.0    # Ia
]

x_0_true = [0.0, 0.0, 400.0]

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

x[:, 1] = x_0_true
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

# Run a staged PMMH sampler.
# Aim for an acceptance ratio of around 20–30% for a random-walk proposal.
PMMH_samples, acceptance_ratio, time_sampling = PMMHopt.staged_PMMH(u_training, y_training, n_x, K, K_b, k_d, N_init, f_theta, g_theta, sample_x_0, sample_v_theta, log_pdf_w_theta, log_pdf_theta, theta_init, theta_cov, T_chunk, K_stage, alpha; regularizer=regularizer, K_adapt=10)
# @load "PMMH_samples.jld2" PMMH_samples

# Formulate the optimal control problem (OCP) using the PMMH samples.

# Revenue from selling the crops.
# The price for the crop is set well above current prices as vertical farming is not yet economically competitive.
price_crop = 10000 # selling price of crop in €/kg
HI = 0.68 # harvest index (proportion of total biomass that is harvestable)
revenue(mB) = price_crop * HI * mB # revenue as a function of biomass in €/m²

# Costs: heating, cooling, radiation, and irrigation.
price_kwh = 0.14 # price per kWh in €
price_MJ = (1 / 3.6) * price_kwh # price per MJ in €

heat_capacity_air = 1.2e-3  # volumetric heat capacity of air in MJ/m³/K
theta_ambient = 10.0  # ambient temperature in °C
theta_max = 35.0  # maximum temperature in °C
c_theta = heat_capacity_air .* price_MJ ./ (theta_max .- theta_ambient) # coefficient for heating/cooling cost
cost_heating(theta) = c_theta .* (theta .- theta_ambient) # heating cost as a function of air temperature in €/m²

cost_radiation(R) = price_MJ * R # cost of radiation as a function of radiation in €/m²

c_d = 0.02 # cost coefficient for irrigation cost
cost_irrigation(D) = c_d .* (D .- 1) .^ 2 # cost of irrigation as a function of the relative level of drought in €/m²

# Objective: maximize profit.
profit(u, x, y) = revenue(x[1, end]) .- sum(cost_heating(u[1, :]) .- cost_radiation(u[3, :]) .- cost_irrigation(u[2, :])) # profit in €/m²
J(u, x, y) = -profit(u, x, y) # cost function to be minimized (negative profit)

# Scenario dependent constraints for u, x, and y.
h_scenario(u, x, y) = 0.0

# Scenario independent constraints for the inputs u.
h_u(u) = [
    u[1, :] .- 35.0; # maximum temperature
    0.0 .- u[1, :]; # minimum temperature
    u[2, :] .- 1.0; # maximum drought index
    0.0 .- u[2, :]; # minimum drought index
    u[3, :] .- 35.0; # maximum radiation
    0.0 .- u[3, :] # minimum radiation
]

# Parameters for the OCP.
H = 30 # time horizon in days
K_pre_solve = 10 # number of samples used to pre-solve the OCP to get a good initial guess

# Ipopt options
Ipopt_options = Dict("max_iter" => 100000, "tol" => 1e-6, "hsllib" => HSL_jll.libhsl_path, "linear_solver" => "ma57", "print_timing_statistics" => "yes") # "hessian_approximation" => "limited-memory", "nlp_scaling_method" => "gradient-based", "mu_strategy" => "adaptive"

# Start optimization.
# u_opt, x_opt, y_opt, J_opt = PMMHopt.solve_PMMH_OCP(PMMH_samples, n_y, f_theta, g_theta, sample_v_theta, sample_w_theta, H, J, h_scenario, h_u; K_pre_solve=K_pre_solve, solver_opts=Ipopt_options)[1:4]
U_init = zeros(n_u, H) # initial guess for the input trajectory
U_opt, X_opt, Y_opt, J_opt = PMMHopt.solve_PMMH_OCP(PMMH_samples, n_y, f_theta, g_theta, sample_v_theta, sample_w_theta, H, J, h_scenario, h_u; U_init=U_init, solver_opts=Ipopt_options)[1:4]

# Helper function to simulate the system forward using different input trajectories and noise realizations.
function simulate_system(f, g, x_t, u, V, W)
    H = size(u, 2)
    n_x = length(x_t)
    n_y = size(W, 1)

    x = Array{Float64}(undef, n_x, H)
    y = Array{Float64}(undef, n_y, H)

    x[:, 1] = x_t

    for t = 2:H
        x[:, t] = f(x[:, t-1], u[:, t-1]) + V[:, t-1]
    end
    for t = 1:H
        y[:, t] = g(x[:, t], u[:, t]) + W[:, t]
    end
    return x, y
end

# Generate noise realizations for the simulations.
V = sample_v_theta(theta_true, H)
W = sample_w_theta(theta_true, H)

# Simulate the system forward using the optimized input trajectory.
x_true_opt, y_true_opt = simulate_system(f_true, g_true, x_test[:, 1], U_opt, V, W)
profit_opt = profit(U_opt, x_true_opt, y_true_opt)

# Plot predictions.
plot_predictions(Y_opt, y_true_opt)
