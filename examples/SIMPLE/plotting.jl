"""
    plot_history(state)

Plots the simulation history for a SIMPLE_state object.

# Arguments
- `state`: SIMPLE state object
"""
function plot_history(state::SIMPLE_state)
    t = 1:length(state.history["mB_hist"])  # Time axis (days)

    # Plot states
    p1 = plot(t, state.history["mB_hist"], xlabel="Days", ylabel="Biomass Density (kg/m²)", title="Biomass Density Over Time", lw=2)
    p2 = plot(t, state.history["tau_hist"], xlabel="Days", ylabel="Cumulative Temperature (°C d)", title="Cumulative Temperature Over Time", lw=2)
    p3 = plot(t, state.history["I50B_hist"], xlabel="Days", ylabel="Canopy Senescence (°C d)", title="Canopy Senescence Over Time", lw=2)

    # Plot inputs
    p4 = plot(t, state.history["theta_hist"], xlabel="Days", ylabel="Temperature (°C)", title="Temperature Over Time", lw=2)
    p5 = plot(t, state.history["D_hist"], xlabel="Days", ylabel="Level of Drought (%)", title="Level of Drought Over Time", lw=2)
    p6 = plot(t, state.history["R_hist"], xlabel="Days", ylabel="Radiation (MJ/m²/d)", title="Radiation Over Time", lw=2)
    p7 = plot(t, state.history["CO2_hist"], xlabel="Days", ylabel="CO2 Concentration (ppm)", title="CO2 Concentration Over Time", lw=2)

    display(p1)
    display(p2)
    display(p3)
    display(p4)
    display(p5)
    display(p6)
    display(p7)
end