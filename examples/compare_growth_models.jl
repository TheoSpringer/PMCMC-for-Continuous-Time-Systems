include("SIMPLE/SIMPLE.jl")
include("TOMGRO/TOMGRO.jl")

using .SIMPLE
using .TOMGRO

# Parameters
days = 100  # number of days

# Input data for both models
T = fill(25.0, days)
D = fill(0.0, days)
R = fill(25.0, days)
PPFD = TOMGRO.radiation2ppfd(R)
CO2 = fill(700.0, days)

######### TOMGRO Simulation #########
parameters_TOMGRO, init_state_TOMGRO = TOMGRO.reset()
inputs_TOMGRO = [TOMGRO.TOMGRO_input(t, ppfd, co2) for (t, ppfd, co2) in zip(T, PPFD, CO2)]

states_TOMGRO = Vector{TOMGRO.TOMGRO_state}(undef, days)
states_TOMGRO[1] = init_state_TOMGRO

# Simulate TOMGRO for each day.
for d in 2:days
    states_TOMGRO[d] = TOMGRO.step(parameters_TOMGRO, states_TOMGRO[d-1], inputs_TOMGRO[d-1])
end

# Plot TOMGRO results.
TOMGRO.plot_history(states_TOMGRO, inputs_TOMGRO)

######### SIMPLE Simulation #########
parameters_SIMPLE, init_state_SIMPLE = SIMPLE.reset()
inputs_SIMPLE = [SIMPLE.SIMPLE_input(t, d, r, co2) for (t, d, r, co2) in zip(T, D, R, CO2)]

states_SIMPLE = Vector{SIMPLE.SIMPLE_state}(undef, days)
states_SIMPLE[1] = init_state_SIMPLE

# Simulate SIMPLE for each day.
for d in 2:days
    states_SIMPLE[d] = SIMPLE.step(parameters_SIMPLE, states_SIMPLE[d-1], inputs_SIMPLE[d-1])
end

# Plot SIMPLE results.
SIMPLE.plot_history(states_SIMPLE, inputs_SIMPLE)