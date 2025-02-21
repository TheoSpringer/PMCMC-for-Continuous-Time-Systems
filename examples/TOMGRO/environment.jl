"""
    TOMGRO_reset(parameters=nothing)

Resets the TOMGRO simulation environment to its initial state.

# Arguments
- `parameters`: if provided, it will override the default parameters.

# Returns
- TOMGRO state
- TOMGRO parameters
"""
function TOMGRO_reset(parameters=nothing)
    if parameters === nothing
        parameters = default_parameters()
    end

    state = TOMGRO_state(
        N_init, LAI_init, W_init, Wm_init, Wf_init,
        Dict(
            "N_hist" => Float64[], "LAI_hist" => Float64[], "W_hist" => Float64[],
            "Wm_hist" => Float64[], "Wf_hist" => Float64[], "Td_hist" => Float64[],
            "PPFDd_hist" => Float64[], "CO2_hist" => Float64[]
        )
    )
    return state, parameters
end

"""
    TOMGRO_step!(state::TOMGRO_state, parameters::TOMGRO_parameters, Td, PPFDd, CO2)

Advances the simulation by one day given temperature, PPFD and CO₂. The inputs are assumed to be constant over the day.

# Arguments
- `state`: current TOMGRO state
- `parameters`: TOMGRO parameters
- `Td`: temperature
- `PPFDd`: photosynthetic photon flux density
- `CO2`: CO₂ concentration

# Returns
- updated TOMGRO state
"""
function TOMGRO_step!(state::TOMGRO_state, parameters::TOMGRO_parameters, Td, PPFDd, CO2)
    # Extract state variables
    N, LAI, W, Wm, Wf = state.N, state.LAI, state.W, state.Wm, state.Wf

    # Temperature is assumed to be constant over the day
    Tdaytime = Td

    # dN/dt
    fN_ = fN(Td)
    dNdt_ += dNdt(fN_, parameters)

    # d(LAI)/dt
    lambda_ = lambda(Td)
    dLAIdt_ = dLAIdt(LAI, 3.10, N, lambda_, dNdt_, parameters)

    # dWfdt
    fR_ = fR(N)
    LFmax_ = LFmax(inCO2[i], parameters)
    PGRED_ = PGRED(Td)
    Pg_ = Pg(LFmax_, PGRED_, PPFDd, LAI, parameters)
    Rm_ = Rm(Td, W, Wm)
    GRnet_ = GRnet(Pg_, Rm_, fR_)
    fF_ = fF(Td)
    g_ = g(Tdaytime, parameters)
    dWfdt_ = dWfdt(GRnet_, fF_, N, g_, parameters)

    # dWdt
    dWdt_ = dWdt(LAI, dWfdt_, GRnet_, 3.10, dNdt_, parameters)

    # dWmdt
    Df_ = Df(Td)
    dWmdt_ = dWmdt(Df_, Wf, Wm, N)

    # Update state variables
    state.N += dNdt_
    state.LAI += dLAIdt_
    state.W += dWdt_
    state.Wm += dWmdt_
    state.Wf += dWfdt_

    # Store history
    push!(state.history["N_hist"], state.N)
    push!(state.history["LAI_hist"], state.LAI)
    push!(state.history["W_hist"], state.W)
    push!(state.history["Wf_hist"], state.Wf)
    push!(state.history["Wm_hist"], state.Wm)
    push!(state.history["Td_hist"], Td)
    push!(state.history["PPFDd_hist"], PPFDd)
    push!(state.history["CO2_hist"], CO2)

    return state
end