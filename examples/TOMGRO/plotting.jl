"""
    plot_history(state)

Plots the simulation history for a TOMGRO_state object.

# Arguments
- `state`: TOMGRO state object
"""
function plot_history(state::TOMGROState, climate_data=nothing)
    t = 1:length(state.history["N_hist"])  # Time axis (days)

    # Plot states
    p1 = plot(t, state.history["N_hist"], xlabel="Days", ylabel="Number of Nodes on Mainstem", title="Number of Nodes Over Time", lw=2)
    p2 = plot(t, state.history["LAI_hist"], xlabel="Days", ylabel="Leaf Area Index (m²/m²)", title="LAI Over Time", lw=2)
    p3 = plot(t, state.history["W_hist"], xlabel="Days", ylabel="Above-Ground Dry Weight (g/m²)", title="Above-Ground Dry Weight Over Time", lw=2)
    p4 = plot(t, state.history["Wf_hist"], xlabel="Days", ylabel="Fruit Dry Weight (g/m²)", title="Total Fruit Dry Weight Over Time", lw=2)
    p5 = plot(t, state.history["Wm_hist"], xlabel="Days", ylabel="Mature Fruit Dry Weight (g/m²)", title="Mature Fruit Dry Weight Over Time", lw=2)

    # Plot inputs
    p6 = plot(t_climate, state.history["Td_hist"], xlabel="Days", ylabel="Temperature (°C)", title="Temperature Over Time", lw=2)
    p7 = plot(t_climate, state.history["PPFDd_hist"], xlabel="Days", ylabel="PPFD (μmol/m²/s)", title="PPFD Over Time", lw=2)
    p8 = plot(t_climate, state.history["CO2_hist"], xlabel="Days", ylabel="CO2 Concentration (ppm)", title="CO2 Concentration Over Time", lw=2)

    display(p1)
    display(p2)
    display(p3)
    display(p4)
    display(p5)
    display(p6)
    display(p7)
    display(p8)
end