"""
# TOMGRO Simulation Module

This module simulates the growth of tomato plants based on environmental factors such as temperature, radiation, and CO₂ concentration. 
It models various aspects of plant development, including node formation, leaf area expansion, dry matter accumulation, and fruit development.

## References
- [Code Reference](https://gist.github.com/gyosit/abeab4e595d7ddcd65b55c1270d240c8)
- Jones (1999) *Reduced state-variable tomato growth model*
- Jones (1991) *A dynamic tomato growth and yield model (TOMGRO)*
- Dimokas (2009) *Calibration and validation of a biological model to simulate the development and production of tomatoes in Mediterranean greenhouses during winter period*
- Heuvelink (1994) *Dry-matter partitioning in a tomato crop: Comparison of two simulation models*
"""
module TOMGRO

export reset, step!, plot_history

"""
## Model Parameters
Constants used in the tomato growth model.

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
const Nm = 0.495
const Nb = 13
const sigma = 0.041
const beta = 0.22
const Vmax = 6
const Qe = 0.09
const tau = 0.12
const K = 0.61
const CE = 0.74
const T_CRIT = 24
const alpha_F = 0.95
const v = 0.24
const LAImax = 6.0

"""
## Initial State
Initial values for model state variables.

- `N`: number of nodes on mainstem
- `LAI`: leaf area index
- `W`: above-ground dry weight
- `Wm`: mature fruit dry weight
- `Wf`: total fruit dry weight
"""
const N_init = 10.0
const LAI_init = 0.05
const W_init = 0.0
const Wm_init = 0.0
const Wf_init = 0.0

"""
## TOMGRO State

Holds the state of the TOMGRO simulation, including historical data.

- `N`: number of nodes on mainstem
- `LAI`: leaf area index
- `W`: above-ground dry weight
- `Wm`: mature fruit dry weight
- `Wf`: total fruit dry weight
"""
mutable struct TOMGRO_state
    N::Float64
    LAI::Float64
    W::Float64
    Wm::Float64
    Wf::Float64
    history::Dict{String,Vector{Float64}}
end

# Include dependencies
include("dynamics.jl")
include("environment.jl")
include("plotting.jl")

end