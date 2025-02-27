"""
# SIMPLE Simulation Module

This module simulates the growth of plants based on environmental factors such as temperature and radiation.

## References
- Zhao (2019) *A SIMPLE crop model*
- Daniels (2023) *Optimal Control for Indoor Vertical Farms Based on Crop Growth*
- Woli (2012) *Agricultural reference index for drought (ARID)*
"""
module SIMPLE

"""
## Model Parameters
Holds the parameters of the SIMPLE simulation.

- `tau_sum`: cumulative temperature requirement from sowing to maturity
- `HI`: potential harvest index
- `Ia`: cumulative temperature requirement for leaf area development to intercept 50 % of radiation
- `Ib`: cumulative temperature till maturity to reach 50 % radiation interception due to leaf senescence
- `theta_base`: base temperature for phenology development and growth
- `theta_opt`: optimal temperature for biomass growth
- `RUE`: Radiation use efficiency
- `Iheat`: maximum daily reduction in I50B due to heat stress
- `Iwater`: maximum daily reduction in I50B due to drought stress
- `theta_heat`: threshold temperature to start accelerating senescence from heat stress
- `theta_ext`: extreme temperature threshold when RUE becomes 0 due to heat stress
- `Sco2`: relative increase in RUE per ppm elevated CO2 above 350 ppm
- `Swater`: sensitivity of RUE to drought stress
- `Rmax`: maximum fraction of radiation interception
"""
mutable struct SIMPLE_parameters
    tau_sum::Float64
    HI::Float64
    Ia::Float64
    Ib::Float64
    theta_base::Float64
    theta_opt::Float64
    RUE::Float64
    Iheat::Float64
    Iwater::Float64
    theta_heat::Float64
    theta_ext::Float64
    Sco2::Float64
    Swater::Float64
    Rmax::Int
end

"""
    default_parameters()

Returns the default SIMPLE model parameters (tomato crop, SunnySD cultivar).
"""
function default_parameters()
    return TOMGROParameters(
        2800,   # tau_sum
        0.68,   # HI
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
        0.95   # Rmax
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

# Include dependencies
include("dynamics.jl")
include("environment.jl")
include("plotting.jl")
end