module ScenarioBase

export PMCMC_sample

"""
    PMCMC_sample

Container for samples from a particle MCMC algorithm.
"""
mutable struct PMCMC_sample
    theta::Array{Float64} # parameters
    x_m1::Array{Float64} # states in the last timestep of the training dataset (t-1)
    w_m1::Array{Float64} # weights in the last timestep of the training dataset (t-1)
    u_m1::Array{Float64} # input in the last timestep of the training dataset (t-1) - required to make predictions
    x_0::Array{Float64} # states in the first timestep of the training dataset (t=0)
    w_0::Array{Float64} # weights in the first timestep of the training dataset (t=0)
end

end