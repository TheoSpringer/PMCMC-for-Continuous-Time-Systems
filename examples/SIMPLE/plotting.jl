"""
    plot_history(state)

Plots the simulation history for a SIMPLE_state object.

# Arguments
- `state`: SIMPLE state
"""
function plot_history(state::SIMPLE_state, climate_data=nothing)
    t = 1:length(state.history["N_hist"])  # Time axis (days)

    # Plot states
    p1 = plot(t, state.history["mB_hist"], xlabel="Days", ylabel="Biomass Density (kg/m²)", title="Number of Nodes Over Time", lw=2)
    p2 = plot(t, state.history["tau_hist"], xlabel="Days", ylabel="Cumulative Temperature (°C d)", title="LAI Over Time", lw=2)
    p3 = plot(t, state.history["I50B_hist"], xlabel="Days", ylabel="Canopy Senescence (°C d)", title="Above-Ground Dry Weight Over Time", lw=2)

    # Plot inputs
    p4 = plot(t_climate, state.history["theta_hist"], xlabel="Days", ylabel="Temperature (°C)", title="Temperature Over Time", lw=2)
    p5 = plot(t_climate, state.history["D_hist"], xlabel="Days", ylabel="Level of Drought (%)]", title="PPFD Over Time", lw=2)
    p6 = plot(t_climate, state.history["R_hist"], xlabel="Days", ylabel="Radiation (MJ/(m^2 d))", title="CO2 Concentration Over Time", lw=2)
    p7 = plot(t_climate, state.history["CO2_hist"], xlabel="Days", ylabel="CO2 Concentration (ppm)", title="CO2 Concentration Over Time", lw=2)

    display(p1)
    display(p2)
    display(p3)
    display(p4)
    display(p5)
    display(p6)
    display(p7)
    display(p8)
end