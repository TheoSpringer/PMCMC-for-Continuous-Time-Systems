"Core module of the PMMHopt algorithm. Contains all necessary functions such as the particle marginal Metropolis Hastings sampler and scenario optimization."
module PMMHopt
using LinearAlgebra
using Statistics
using Distributions
using StatsBase
using Plots
using LaTeXStrings
using Printf
using JuMP
using Ipopt
export PMMH_sample, particle_filter, particle_MMH, staged_PMMH, compute_ess, epsilon, solve_PMMH_OCP, solve_PMMH_OCP_greedy_guarantees, test_prediction, plot_predictions, plot_autocorrelation, plot_parameter_trace, plot_parameter_pdf

# Struct for the samples of the PMMH algorithm
mutable struct PMMH_sample
    theta::Array{Float64} # parameters
    x_m1::Array{Float64} # states in the last timestep of the training dataset (t-1)
    w_m1::Array{Float64} # weights in the last timestep of the training dataset (t-1)
    u_m1::Array{Float64} # input in the last timestep of the training dataset (t-1) - required to make predictions
    x_0::Array{Float64} # states in the first timestep of the training dataset (t=0)
    w_0::Array{Float64} # weights in the first timestep of the training dataset (t=0)
end

include("PMMH.jl")
include("optimal_control.jl")
include("plotting.jl")
end