"Core module of PMCMCopt. Contains all necessary functions such as the particle marginal Metropolis Hastings sampler and the scenario optimization."
module PMCMCopt
using LinearAlgebra
using Statistics
using Distributions
using Random
using StatsBase
using Base.Threads
using Plots
using LaTeXStrings
using Printf
using JuMP
using Symbolics
import MathOptInterface as MOI
using Ipopt
using SparseArrays
using ADTypes
using DifferentiationInterface
using SparseConnectivityTracer
using SparseMatrixColorings
using ForwardDiff
using ReverseDiff
using Enzyme

export PMCMC_sample, particle_filter, adapt_N, particle_MMH, staged_PMMH, compute_ess, compute_gelman_rubin, epsilon, solve_PMMH_OCP, solve_PMMH_OCP_greedy_guarantees, test_prediction, plot_predictions, plot_autocorrelation, plot_parameter_trace, plot_parameter_pdf

# Struct for the samples of the PMMH algorithm
mutable struct PMCMC_sample
    theta::Array{Float64} # parameters
    x_m1::Array{Float64} # states in the last timestep of the training dataset (t-1)
    w_m1::Array{Float64} # weights in the last timestep of the training dataset (t-1)
    u_m1::Array{Float64} # input in the last timestep of the training dataset (t-1) - required to make predictions
    x_0::Array{Float64} # states in the first timestep of the training dataset (t=0)
    w_0::Array{Float64} # weights in the first timestep of the training dataset (t=0)
end

include("sampling/helpers.jl")
include("sampling/particle_filter.jl")
include("sampling/particle_MMH.jl")
include("sampling/particle_MMH_blocked.jl")
include("sampling/diagnostics.jl")
include("optimal_control/epsilon.jl")
include("optimal_control/optimal_control.jl")
# include("optimal_control/optimal_control_legacy.jl")
include("optimal_control/evaluator/types.jl")
include("optimal_control/evaluator/indices.jl")
include("optimal_control/evaluator/helpers.jl")
include("optimal_control/evaluator/sparsity.jl")
include("optimal_control/evaluator/evaluator.jl")
include("optimal_control/evaluator/callbacks.jl")
end