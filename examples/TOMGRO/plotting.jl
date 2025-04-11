"""
    plot_history(states, inputs)

Plots the simulation history for a TOMGRO simulation.

# Arguments
- `states`: A vector of `TOMGRO_state` objects. Each state should have fields `N`, `LAI`, `W`, `Wf`, and `Wm`.
- `inputs`: A vector of `TOMGRO_input` objects. Each input should have fields `Td`, `PPFDd`, and `CO2`.

This function extracts the time series from the vector of states and inputs and generates eight plots:
- Five for state variables (Number of Nodes, LAI, Above-Ground Dry Weight, Fruit Dry Weight, and Mature Fruit Dry Weight).
- Three for input variables (Temperature, PPFD, and CO2 Concentration).
"""
function plot_history(states::Vector{TOMGRO_state}, inputs::Vector{TOMGRO_input})
    # Assume that the length of states and inputs corresponds to the number of timesteps.
    t = 1:length(states)

    # Extract state variables from the array of states.
    N_hist = [s.N for s in states]
    LAI_hist = [s.LAI for s in states]
    W_hist = [s.W for s in states]
    Wf_hist = [s.Wf for s in states]
    Wm_hist = [s.Wm for s in states]

    # Extract input variables from the array of inputs.
    Td_hist = [inp.Td for inp in inputs]
    PPFDd_hist = [inp.PPFDd for inp in inputs]
    CO2_hist = [inp.CO2 for inp in inputs]

    # Create plots for the state variables.
    p1 = plot(t, N_hist, xlabel="Days", ylabel="Number of Nodes on Mainstem",
        title="Number of Nodes Over Time", lw=2)
    p2 = plot(t, LAI_hist, xlabel="Days", ylabel="Leaf Area Index (m²/m²)",
        title="LAI Over Time", lw=2)
    p3 = plot(t, W_hist, xlabel="Days", ylabel="Above-Ground Dry Weight (g/m²)",
        title="Above-Ground Dry Weight Over Time", lw=2)
    p4 = plot(t, Wf_hist, xlabel="Days", ylabel="Fruit Dry Weight (g/m²)",
        title="Total Fruit Dry Weight Over Time", lw=2)
    p5 = plot(t, Wm_hist, xlabel="Days", ylabel="Mature Fruit Dry Weight (g/m²)",
        title="Mature Fruit Dry Weight Over Time", lw=2)

    # Create plots for the input variables.
    p6 = plot(t, Td_hist, xlabel="Days", ylabel="Temperature (°C)",
        title="Temperature Over Time", lw=2)
    p7 = plot(t, PPFDd_hist, xlabel="Days", ylabel="PPFD (μmol/m²/s)",
        title="PPFD Over Time", lw=2)
    p8 = plot(t, CO2_hist, xlabel="Days", ylabel="CO2 Concentration (ppm)",
        title="CO2 Concentration Over Time", lw=2)

    # Display each plot.
    display(p1)
    display(p2)
    display(p3)
    display(p4)
    display(p5)
    display(p6)
    display(p7)
    display(p8)
end