"""
# Tomato Growth Model
This module simulates the growth of tomato plants based on environmental factors such as temperature, radiation, and CO₂ concentration. 
It models various aspects of plant development, including node formation, leaf area expansion, dry matter accumulation, and fruit development.

Reference:
- CODE: https://gist.github.com/gyosit/abeab4e595d7ddcd65b55c1270d240c8
- Jones (1999) "Reduced state-variable tomato growth model"
- Jones (1991) "A dynamic tomato growth and yield model (TOMGRO)"
- Dimokas (2009) "Calibration and validation of a biological model to simulate the development and production of tomatoes in Mediterranean greenhouses during winter period"
- Heuvelink (1994) "Dry-matter partitioning in a tomato crop: Comparison of two simulation models"
"""
module TOMGRO

export reset, step!

"""
# Parameters
Constants used in the tomato growth model.

- `Nm`: Maximum rate of node appearance (at optimal temperatures)
- `Nb`: Coefficient in expolinear equation, projection of linear segment of LAI vs N to horizontal axis
- `sigma`: Maximum leaf area expansion per node, coefficient in expolinear equation
- `beta`: Coefficient in expolinear equation
- `Vmax`: Maximum increase in vegetative tissue d.w. growth per node
- `Qe`: Leaf quantum efficiency
- `tau`: Carbon dioxide use efficiency
- `K`: Light extinction coefficient
- `CE`: Conversion coefficient for assimilated carbon into dry matter
- `T_CRIT`: Mean daytime temperature above which fruit abortion starts
- `alpha_F`: Maximum partitioning of new growth to fruit
- `v`: Transition coefficient governing the shift between vegetative and reproductive growth phases
- `LAImax`: Maximum leaf area index
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
# Initial State
Initial values for model state variables.

- `N`: Number of nodes on mainstem
- `LAI`: Leaf area index
- `W`: Above-ground dry weight
- `Wm`: Mature fruit dry weight
- `Wf`: Total fruit dry weight
"""
const N_init = 10.0
const LAI_init = 0.05
const W_init = 0.0
const Wm_init = 0.0
const Wf_init = 0.0

# Global variables to store simulation state and history
mutable struct TOMGRO_state
    N::Float64
    LAI::Float64
    W::Float64
    Wm::Float64
    Wf::Float64
    history::Dict{String,Vector{Float64}}
end

# Include submodules/files
include("dynamics.jl")
include("environment.jl")

using .dynamics
using .environment

end