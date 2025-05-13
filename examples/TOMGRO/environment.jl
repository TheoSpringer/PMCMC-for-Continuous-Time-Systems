"""
    reset(parameters::Union{Nothing,TOMGRO_parameters}=nothing)

Resets the TOMGRO simulation environment to its initial state.

# Arguments
- `parameters`: if provided, it will override the default parameters.

# Returns
- TOMGRO parameters
- TOMGRO state
"""
function reset(parameters::Union{Nothing,TOMGRO_parameters}=nothing)
    if parameters === nothing
        parameters = default_parameters()
    end

    state = TOMGRO_state(
        N_init, LAI_init, W_init, Wm_init, Wf_init
    )
    return parameters, state
end

"""
    step!(parameters::TOMGRO_parameters, state::TOMGRO_state, input::TOMGRO_input)

Advances the simulation by one day given temperature, PPFD and CO₂. The inputs are assumed to be constant over the day.

# Arguments
- `parameters`: TOMGRO parameters
- `state`: current TOMGRO state
- `input`: current TOMGRO input

# Returns
- updated TOMGRO state
"""
function step!(parameters::TOMGRO_parameters, state::TOMGRO_state, input::TOMGRO_input)
    # Extract state and input variables
    N, LAI, W, Wm, Wf = state.N, state.LAI, state.W, state.Wm, state.Wf
    Td, PPFDd, CO2 = input.Td, input.PPFDd, input.CO2

    # Temperature is assumed to be constant over the day
    Tdaytime = Td

    # dN/dt
    fN_ = fN(Td)
    dNdt_ = dNdt(fN_, parameters)

    # d(LAI)/dt
    lambda_ = lambda(Td)
    dLAIdt_ = dLAIdt(LAI, 3.10, N, lambda_, dNdt_, parameters)

    # dWfdt
    fR_ = fR(N)
    LFmax_ = LFmax(CO2, parameters)
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

    return state
end

"""
    step(parameters::TOMGRO_parameters, state::TOMGRO_state, input::TOMGRO_input)

Advances the simulation by one day given temperature, PPFD and CO₂. The inputs are assumed to be constant over the day.

# Arguments
- `parameters`: TOMGRO parameters
- `state`: current TOMGRO state
- `input`: current TOMGRO input

# Returns
- updated TOMGRO state
"""
function step(parameters::TOMGRO_parameters, state::TOMGRO_state, input::TOMGRO_input)
    # Extract state and input variables
    N, LAI, W, Wm, Wf = state.N, state.LAI, state.W, state.Wm, state.Wf
    Td, PPFDd, CO2 = input.Td, input.PPFDd, input.CO2

    # Temperature is assumed to be constant over the day
    Tdaytime = Td

    # dN/dt
    fN_ = fN(Td)
    dNdt_ = dNdt(fN_, parameters)

    # d(LAI)/dt
    lambda_ = lambda(Td)
    dLAIdt_ = dLAIdt(LAI, 3.10, N, lambda_, dNdt_, parameters)

    # dWfdt
    fR_ = fR(N)
    LFmax_ = LFmax(CO2, parameters)
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
    N_next = N + dNdt_
    LAI_next = LAI + dLAIdt_
    W_next = W + dWdt_
    Wm_next = Wm + dWmdt_
    Wf_next = Wf + dWfdt_

    return TOMGRO_state(N_next, LAI_next, W_next, Wm_next, Wf_next)
end