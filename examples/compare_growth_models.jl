include("SIMPLE/SIMPLE.jl")
include("TOMGRO/TOMGRO.jl")

using .SIMPLE
using .TOMGRO

# Parameters
days = 100 # number of days
T = fill(25.0, days)
D = fill(0.0, days)
R = fill(25.0, days)
PPFD = TOMGRO.radiation2ppfd(R)
CO2 = fill(400.0, days)

# Simulate TOMGRO model
state_TOMGRO, parameters_TOMGRO = TOMGRO.reset()
for d in 1:days
    TOMGRO.step!(state_TOMGRO, parameters_TOMGRO, T[d], PPFD[d], CO2[d])
end

# Plot results
TOMGRO.plot_history(state_TOMGRO)

# Simulate SIMPLE model
state_SIMPLE, parameters_SIMPLE = SIMPLE.reset()
for d in 1:days
    SIMPLE.step!(state_SIMPLE, parameters_SIMPLE, T[d], D[d], R[d], CO2[d])
end

# Plot results
SIMPLE.plot_history(state_SIMPLE)