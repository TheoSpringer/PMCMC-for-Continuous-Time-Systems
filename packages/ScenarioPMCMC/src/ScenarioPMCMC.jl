"Core module of ScenarioPMCMC."
module ScenarioPMCMC
using LinearAlgebra
using Statistics
using Distributions
using Random
using StatsBase
using Base.Threads
using Plots
using LaTeXStrings
using Printf

using ScenarioBase

export particle_filter, adapt_N, particle_MMH, staged_PMMH, compute_ess, compute_gelman_rubin, test_prediction, plot_predictions, plot_autocorrelation, plot_parameter_trace, plot_parameter_pdf

include("helpers.jl")
include("particle_filter.jl")
include("particle_MMH.jl")
include("particle_MMH_blocked.jl")
include("diagnostics.jl")
end