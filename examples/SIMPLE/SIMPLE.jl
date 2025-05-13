"""
# SIMPLE Simulation Module

This module simulates the growth of plants based on environmental factors such as temperature and radiation.

## References
- Zhao (2019) *A SIMPLE crop model*
- Daniels (2023) *Optimal Control for Indoor Vertical Farms Based on Crop Growth*
- Woli (2012) *Agricultural reference index for drought (ARID)*
"""
module SIMPLE
using Plots

export SIMPLE_parameters, SIMPLE_state, SIMPLE_input, reset, step!, get_yield, plot_history

"""
## Model Parameters
Holds the parameters of the SIMPLE simulation. Compared to the original paper, the I50B parameter is not included as it describes the initial state of the simulation and is not a parameter of the dynamics.

- `tau_sum`: cumulative temperature requirement from sowing to maturity
- `Ia`: cumulative temperature requirement for leaf area development to intercept 50 % of radiation
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
- `HI`: harvest index
"""
mutable struct SIMPLE_parameters
    tau_sum::Float64
    Ia::Float64
    theta_base::Float64
    theta_opt::Float64
    RUE::Float64
    Iheat::Float64
    Iwater::Float64
    theta_heat::Float64
    theta_ext::Float64
    Sco2::Float64
    Swater::Float64
    Rmax::Float64
    HI::Float64
end

"""
    default_parameters()

Returns the default SIMPLE model parameters (tomato crop, SunnySD cultivar).
"""
function default_parameters()
    return SIMPLE_parameters(
        2800.0,   # tau_sum
        520.0,    # Ia
        6.0,    # theta_base
        26.0,   # theta_opt
        1.00 * 1e-3,    # RUE
        100.0,  # Iheat
        5.0,    # Iwater
        32.0,   # theta_heat
        45.0,   # theta_ext
        0.07,   # Sco2
        2.5,    # Swater
        0.95,   # Rmax
        0.68   # HI
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
const I50B_init = 400.0 # I50B parameter in paper

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
end

"""
## SIMPLE Input

Holds the input of the SIMPLE simulation.

- `theta`: temperature
- `D`: relative level of drought (ARID index); see Woli (2012) 
- `R`: radiation
- `CO2`: atmospheric CO₂ concentration
"""
mutable struct SIMPLE_input
    theta::Float64
    D::Float64
    R::Float64
    CO2::Float64
end

# Include dependencies
include("dynamics.jl")
include("environment.jl")
include("plotting.jl")
end