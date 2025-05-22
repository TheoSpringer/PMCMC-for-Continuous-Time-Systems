"""
# TOMGRO Simulation Module

This module simulates the growth of tomato plants based on environmental factors such as temperature, radiation, and CO₂ concentration. 
It models various aspects of plant development, including node formation, leaf area expansion, dry matter accumulation, and fruit development.

## References
- [Code Reference](https://gist.github.com/gyosit/abeab4e595d7ddcd65b55c1270d240c8)
- Jones, J. W., A. Kenig, and C. E. Vallejos. "Reduced state–variable tomato growth model." Transactions of the ASAE 42.1 (1999): 255-265.
- Jones, James W., et al. "A dynamic tomato growth and yield model (TOMGRO)." Transactions of the ASAE 34.2 (1991): 663-0672.
- Dimokas, George, Marc Tchamitchian, and Constantin Kittas. "Calibration and validation of a biological model to simulate the development and production of tomatoes in Mediterranean greenhouses during winter period." biosystems engineering 103.2 (2009): 217-227.
- Heuvelink, Egbert, and Nadia Bertin. "Dry-matter partitioning in a tomato crop: comparison of two simulation models." Journal of horticultural science 69.5 (1994): 885-903.
"""
module TOMGRO
using Plots

export TOMGRO_parameters, TOMGRO_state, TOMGRO_input, reset, step!, plot_history, radiation2ppfd

"""
## Model Parameters
Holds the parameters of the TOMGRO simulation.

- `Nm`: maximum rate of node appearance (at optimal temperatures)
- `Nb`: coefficient in expolinear equation, projection of linear segment of LAI vs N to horizontal axis
- `sigma`: maximum leaf area expansion per node, coefficient in expolinear equation
- `beta`: coefficient in expolinear equation
- `Vmax`: maximum increase in vegetative tissue dry weight growth per node
- `Qe`: leaf quantum efficiency
- `tau`: carbon dioxide use efficiency
- `K`: light extinction coefficient
- `CE`: conversion coefficient for assimilated carbon into dry matter
- `T_CRIT`: mean daytime temperature above which fruit abortion starts
- `alpha_F`: maximum partitioning of new growth to fruit
- `v`: transition coefficient governing the shift between vegetative and reproductive growth phases
- `LAImax`: maximum leaf area index
"""
mutable struct TOMGRO_parameters
    Nm::Float64
    Nb::Int
    sigma::Float64
    beta::Float64
    Vmax::Float64
    Qe::Float64
    tau::Float64
    K::Float64
    CE::Float64
    T_CRIT::Float64
    alpha_F::Float64
    v::Float64
    LAImax::Float64
end

"""
    default_parameters()

Returns the default TOMGRO model parameters.
"""
function default_parameters()
    return TOMGRO_parameters(
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

- `N`: number of nodes on mainstem
- `LAI`: leaf area index in m²/m² (ratio of leaf area per ground area)
- `W`: above-ground dry weight in kg/m²
- `Wm`: mature fruit dry weight in kg/m²
- `Wf`: total fruit dry weight in kg/m²
"""
const N_init = 10.0
const LAI_init = 0.05
const W_init = 0.0
const Wm_init = 0.0
const Wf_init = 0.0

"""
## TOMGRO State

Holds the state of the TOMGRO simulation.

- `N`: number of nodes on mainstem
- `LAI`: leaf area index m²/m² (ratio of leaf area per ground area)
- `W`: above-ground dry weight in kg/m²
- `Wm`: mature fruit dry weight in kg/m²
- `Wf`: total fruit dry weight in kg/m²
"""
mutable struct TOMGRO_state
    N::Float64
    LAI::Float64
    W::Float64
    Wm::Float64
    Wf::Float64
end

"""
## TOMGRO Input

Holds the input of the TOMGRO simulation.

- `Td`: temperature in °C
- `PPFDd`: photosynthetic photon flux density (PPFD) in µmol/m²/s
- `CO2`: CO₂ concentration in ppm
"""
mutable struct TOMGRO_input
    Td::Float64
    PPFDd::Float64
    CO2::Float64
end

# Include dependencies
include("dynamics.jl")
include("environment.jl")
include("plotting.jl")

end