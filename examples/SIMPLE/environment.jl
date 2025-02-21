"""
    SIMPLE_reset(parameters=nothing)

Resets the SIMPLE simulation environment to its initial state.

# Arguments
- `parameters`: if provided, it will override the default parameters.

# Returns
- SIMPLE state
- SIMPLE parameters
"""
function SIMPLE_reset(parameters=nothing)
    if parameters === nothing
        parameters = default_parameters()
    end

    state = SIMPLE_state(
        mB_init, tau_init, I50B_init,
        Dict(
            "mB_hist" => Float64[], "tau_hist" => Float64[], "I50B_hist" => Float64[],
            "theta_hist" => Float64[], "D_hist" => Float64[], "R_hist" => Float64[],
            "CO2_hist" => Float64[]
        )
    )
    return state, parameters
end

"""
    SIMPLE_step!(state::SIMPLE_state, parameters::SIMPLE_parameters, theta, D, R, CO2)

Advances the simulation by one day given temperature, relative level of drought and CO₂. The inputs are assumed to be constant over the day.

# Arguments
- `state`: current SIMPLE state
- `parameters`: SIMPLE parameters
- `theta`: temperature
- `D`: relative level of drought (ARID index); see Woli (2012) 
- `R`: radiation
- `CO2`: atmospheric CO₂ concentration

# Returns
- updated SIMPLE state
"""
function SIMPLE_step!(state::SIMPLE_state, parameters::SIMPLE_parameters, theta, D, R, CO2)
    # Extract state variables
    mB, tau, I50B = state.mB, state.tau, state.I50B

    # Update state variables
    fsolar_ = fsolar(tau, I50B, parameters)
    ftemp_ = ftemp(theta, parameters)
    fheat_ = fheat(theta, parameters)
    fco2_ = fco2(CO2, parameters)
    fwater_ = fwater(D, parameters)
    fdrought_ = fdrought(D, parameters)

    state.mB += R * fsolar_ * parameters.RUE * fco2_ * ftemp_ * fdrought_ * smin(fwater_, fheat_)
    state.tau += smax(theta - parameters.theta_base, 0)
    state.I50B += smax(parameters.Iwater * (1 - fwater_), parameters.Iheat * (1 - fheat_))

    # Store history
    push!(state.history["mB_hist"], state.mB)
    push!(state.history["tau_hist"], state.tau)
    push!(state.history["I50B_hist"], state.I50B)
    push!(state.history["theta_hist"], theta)
    push!(state.history["D_hist"], D)
    push!(state.history["R_hist"], R)
    push!(state.history["CO2_hist"], CO2)

    return state
end

"""
    get_yield(state::SIMPLE_state, parameters::SIMPLE_parameters)

Compute the final yield.

# Arguments
- `state`: current SIMPLE state
- `parameters`: SIMPLE parameters

# Returns
- yield
"""
function get_yield(state::SIMPLE_state, parameters::SIMPLE_parameters)
    yield = state.mB * parameters.HI
    return yield
end