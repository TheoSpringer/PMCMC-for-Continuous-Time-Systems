"Core module of ScenarioOCP."
module ScenarioOCP
using LinearAlgebra
using StatsBase
using Base.Threads
using Printf
using JuMP
import MathOptInterface as MOI
using Ipopt
using SparseArrays
using ADTypes
using DifferentiationInterface
using SparseConnectivityTracer
using SparseMatrixColorings
using ForwardDiff
using ReverseDiff

using ScenarioBase

export epsilon, solve_PMCMC_OCP, solve_PMCMC__OCP_greedy_guarantees

include("epsilon.jl")
include("optimal_control.jl")
# include("optimal_control_legacy.jl")
include("types.jl")
include("indices.jl")
include("helpers.jl")
include("sparsity.jl")
include("evaluator.jl")
include("callbacks.jl")

end