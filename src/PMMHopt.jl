"Core module of the PMMHopt algorithm. Contains all necessary functions such as the particle marginal Metropolis Hastings sampler and scenario optimization."
module PMMHopt

export PMMH_sample, particle_MMH, particle_filter, test_prediction, plot_predictions, plot_autocorrelation, epsilon, solve_PMMH_OCP, solve_PMMH_OCP_greedy_guarantees

# Struct for the samples of the PMMH algorithm
mutable struct PMMH_sample
    theta::Array{Float64} # parameters
    x_m1::Array{Float64} # corresponding states in the last timestep of the training dataset (t=t_0-1)
    w_m1::Array{Float64} # weights in the last timestep of the training dataset (t=t_0-1)
    u_m1::Array{Float64} # input in the last timestep of the training dataset (t=t_0-1) - required to make predictions
end

include("PMMH.jl")
include("optimal_control.jl")
end