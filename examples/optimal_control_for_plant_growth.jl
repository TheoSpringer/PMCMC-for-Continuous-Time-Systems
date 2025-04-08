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

# Learning parameters
K = 200 # number of PMMH samples per stage
k_d = 50 # number of samples to be skipped to decrease correlation (thinning)
K_b = 1000 # length of burn-in period for each stage
N = 30 # number of particles of the particle filter

# Number of states, etc.
n_x = 3 # number of states
n_u = 2 # number of control inputs
n_y = 2 # number of outputs

# State-space prior and proposal distribution.

# Initial guess for model parameters

# Normally distributed initial state

# Define measurement model - assumed to be known (without loss of generality).
# Make sure that g(x, u) is defined in vectorized form, i.e., g(zeros(n_x, N), zeros(n_u, N)) should return a matrix of dimension (n_y, N).
g(x, u) = [1 0] * x # observation function
R = 0.1 # variance of zero-mean Gaussian measurement noise

# Parameters for data generation
D = 50 # number of days for training
D_test = 50  # number of days used for testing (via forward simulation - see below)
D_all = D + D_test

# Generate training data.


# Unknown system

# Input trajectory used to generate training and test data
T = fill(25.0, days)
D = fill(0.0, days)
R = fill(25.0, days)
PPFD = TOMGRO.radiation2ppfd(R)
CO2 = fill(400.0, days)

# Generate data by forward simulation.
state_SIMPLE, parameters_SIMPLE = SIMPLE.reset()
for d in 1:d_training
    SIMPLE.step!(state_SIMPLE, parameters_SIMPLE, T[d], D[d], R[d], CO2[d])
end

# Split data into training and test data.
u_training = u[:, 1:T]
x_training = x[:, 1:T+1]
y_training = y[:, 1:T]

u_test = u[:, T+1:end]
x_test = x[:, T+1:end]
y_test = y[:, T+1:end]

# Plot data.
# plot(Array(1:T_all), u[1,:], label="input", lw=2, legend=:topright);
# plot!(Array(1:T_all), y[1,:], label="output", lw=2);
# xlabel!("t");
# ylabel!("u | y");

# Learn models.
particle_MMH(u, y, n_x, K, K_b, k_d, N, f_theta::Function, g_theta::Function, sample_v_theta::Function, log_pdf_w_theta::Function, pdf_theta::Function, propose_theta::Function, theta_init, sample_x_init::Function)

time_sampling = time() - sampling_timer

# Set up OCP.