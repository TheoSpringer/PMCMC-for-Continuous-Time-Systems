"""
    plot_history(states, inputs)

Plots the simulation history.

# Arguments
- `states`: A vector of SIMPLE_state objects.
- `inputs`: A vector of SIMPLE_input objects.

It constructs time series for the state variables (mB, tau, I50B) and the input variables (theta, D, R, CO2)
using the data from each timestep.
"""
function plot_history(states::Vector{SIMPLE_state}, inputs::Vector{SIMPLE_input})
    # Time axis (assumes the same number of states and inputs)
    t = 1:length(states)

    # Extract state variables from the states array
    mB_hist = [s.mB for s in states]
    tau_hist = [s.tau for s in states]
    I50B_hist = [s.I50B for s in states]

    # Extract input variables from the inputs array
    theta_hist = [inp.theta for inp in inputs]
    D_hist = [inp.D for inp in inputs]
    R_hist = [inp.R for inp in inputs]
    CO2_hist = [inp.CO2 for inp in inputs]

    # Plot state trajectories
    p1 = plot(t, mB_hist, xlabel="Days", ylabel="Biomass Density (kg/m²)",
        title="Biomass Density Over Time", lw=2)
    p2 = plot(t, tau_hist, xlabel="Days", ylabel="Cumulative Temperature (°C d)",
        title="Cumulative Temperature Over Time", lw=2)
    p3 = plot(t, I50B_hist, xlabel="Days", ylabel="Canopy Senescence",
        title="Canopy Senescence Over Time", lw=2)

    # Plot input trajectories
    p4 = plot(t, theta_hist, xlabel="Days", ylabel="Temperature (°C)",
        title="Temperature Over Time", lw=2)
    p5 = plot(t, D_hist, xlabel="Days", ylabel="Level of Drought (%)",
        title="Level of Drought Over Time", lw=2)
    p6 = plot(t, R_hist, xlabel="Days", ylabel="Radiation (MJ/m²/d)",
        title="Radiation Over Time", lw=2)
    p7 = plot(t, CO2_hist, xlabel="Days", ylabel="CO2 Concentration (ppm)",
        title="CO2 Concentration Over Time", lw=2)

    # Display each plot
    display(p1)
    display(p2)
    display(p3)
    display(p4)
    display(p5)
    display(p6)
    display(p7)
end