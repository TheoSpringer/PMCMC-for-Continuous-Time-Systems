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

"""
    get_biomass(state::TOMGRO_state)
Returns the (fresh) biomass in kg/m² used by the SIMPLE model.

# Arguments
- `state`: current TOMGRO state

# Returns
- biomass in kg/m²
"""
function get_biomass(state::TOMGRO_state)
    # Estimate total dry weight including roots from above-ground dry weight.
    # The root-to-shoot ratio depends on species and conditions (e.g., drought).
    # The value of 0.10 is reported in:
    #   Thwe, Aye Aye, et al. "Dynamic shoot and root growth at different developmental stages of tomato (Solanum lycopersicum Mill.) under acute ozone stress." Scientia Horticulturae 150 (2013): 317-325.
    RS_ratio = 0.10
    W_total_dry_g = (1 + RS_ratio) * state.W # total dry weight including roots

    # Convert total dry weight from g/m² to kg/m².
    W_total_dry_kg = W_total_dry_g / 1000.0

    # The fresh weight can be estimated from dry weight using the dry matter content (DMC).
    # The value of 0.177 is reported in 
    #   Ventura, Myriam Rodrigue, M. C. Pieltain, and J. I. R. Castanon. "Evaluation of tomato crop by-products as feed for goats." Animal Feed Science and Technology 154.3-4 (2009): 271-275.
    DMC = 0.177

    # Convert dry weight to fresh weight.
    W_total_fresh = W_total_dry_kg / DMC  # fresh weight in g/m²

    return W_total_fresh
end