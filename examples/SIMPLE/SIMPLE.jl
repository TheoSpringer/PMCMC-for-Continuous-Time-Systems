"""
# SIMPLE Simulation Module

This module simulates the growth of plants based on environmental factors such as temperature and radiation.

## References
- Zhao (2019) *A SIMPLE crop model*
- Daniels (2023) *Optimal Control for Indoor Vertical Farms Based on Crop Growth*
"""
module SIMPLE

"""
## Model Parameters
Holds the parameters of the TOMGRO simulation.

- `Nm`: maximum rate of node appearance (at optimal temperatures)
- `Nb`: coefficient in expolinear equation, projection of linear segment of LAI vs N to horizontal axis
- `sigma`: maximum leaf area expansion per node, coefficient in expolinear equation
- `beta`: coefficient in expolinear equation
- `Vmax`: maximum increase in vegetative tissue d.w. growth per node
- `Qe`: leaf quantum efficiency
- `tau`: carbon dioxide use efficiency
- `K`: light extinction coefficient
- `CE`: conversion coefficient for assimilated carbon into dry matter
- `T_CRIT`: mean daytime temperature above which fruit abortion starts
- `alpha_F`: maximum partitioning of new growth to fruit
- `v`: transition coefficient governing the shift between vegetative and reproductive growth phases
- `LAImax`: maximum leaf area index
"""
mutable struct SIMPLE_parameters
    Eru::Float64
    Rmax::Int
    Sco2::Float64
    theta_base::Float64
    theta_opt::Float64
    theta_heat::Float64
    theta_ext::Float64
    Iwater::Float64
    Iheat::Float64
    Swater::Float64
    Ia::Float64
    Ib::Float64
    tausum::Float64
    CO2::Float64
end

"""
    default_parameters()

Returns the default TOMGRO model parameters.
"""
function default_parameters()
    return TOMGROParameters(
        0.495,  # Nm
        13,     # Nb
        0.041,  # sigma
        0.22,   # beta
        6.0,    # Vmax
        0.09,   # Qe
        0.12,   # tau
        0.61,   # K
        0.74,   # CE
        24.0,   # T_CRIT
        0.95,   # alpha_F
        0.24,   # v
        6.0     # LAImax
    )
end

"""
## Initial State
Initial values for model state variables.

- `mB`: biomass
- `tau`: cumulative temperature
- `I50B`: leaf senescence
"""
const mB_init = 0.0
const tau_init = 0.0
const I50B_init = 50.0


"""
## SIMPLE State

Holds the state of the SIMPLE simulation, including historical data.

- `mB`: biomass
- `tau`: cumulative temperature
- `I50B`: leaf senescence
"""
mutable struct SIMPLE_state
    mB::Float64
    tau::Float64
    I50B::Float64
    history::Dict{String,Vector{Float64}}
end

end