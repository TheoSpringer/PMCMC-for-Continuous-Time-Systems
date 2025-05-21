"""
    reset(parameters=nothing)

Resets the SIMPLE simulation environment to its initial state.

# Arguments
- `parameters`: if provided, it will override the default parameters.

# Returns
- SIMPLE parameters
- SIMPLE state
"""
function reset(parameters::Union{Nothing,SIMPLE_parameters}=nothing)
    if parameters === nothing
        parameters = default_parameters()
    end

    state = SIMPLE_state(
        mB_init, tau_init, I50B_init
    )
    return parameters, state
end

"""
    step!(parameters::SIMPLE_parameters, state::SIMPLE_state, input::SIMPLE_input)

Advances the simulation by one day given inputs (temperature, relative level of drought and CO₂). The inputs are assumed to be constant over the day.

# Arguments
- `parameters`: SIMPLE parameters
- `state`: current SIMPLE state
- `input`: current SIMPLE input

# Returns
- updated SIMPLE state
"""
function step!(parameters::SIMPLE_parameters, state::SIMPLE_state, input::SIMPLE_input)
    # Extract state and input variables
    mB, tau, I50B = state.mB, state.tau, state.I50B
    theta, D, R, CO2 = input.theta, input.D, input.R, input.CO2

    # Update state variables
    fsolar_ = fsolar(tau, I50B, parameters)
    ftemp_ = ftemp(theta, parameters)
    fheat_ = fheat(theta, parameters)
    fco2_ = fco2(CO2, parameters)
    fwater_ = fwater(D, parameters)
    fdrought_ = fdrought(D, parameters)

    state.mB += R * fsolar_ * parameters.RUE * fco2_ * ftemp_ * fdrought_ * smin(fwater_, fheat_)
    state.tau += smax(theta - parameters.theta_base, 0.0)
    state.I50B += smax(parameters.Iwater * (1 - fwater_), parameters.Iheat * (1 - fheat_))

    return state
end

"""
    step(parameters::SIMPLE_parameters, state::SIMPLE_state, input::SIMPLE_input)

Advances the simulation by one day given inputs (temperature, relative level of drought and CO₂). The inputs are assumed to be constant over the day.

# Arguments
- `parameters`: SIMPLE parameters
- `state`: current SIMPLE state
- `input`: current SIMPLE input

# Returns
- new SIMPLE state
"""
function step(parameters::SIMPLE_parameters, state::SIMPLE_state, input::SIMPLE_input)
    # Extract state and input variables
    mB, tau, I50B = state.mB, state.tau, state.I50B
    theta, D, R, CO2 = input.theta, input.D, input.R, input.CO2

    # Update state variables
    fsolar_ = fsolar(tau, I50B, parameters)
    ftemp_ = ftemp(theta, parameters)
    fheat_ = fheat(theta, parameters)
    fco2_ = fco2(CO2, parameters)
    fwater_ = fwater(D, parameters)
    fdrought_ = fdrought(D, parameters)

    mB_next = mB + R * fsolar_ * parameters.RUE * fco2_ * ftemp_ * fdrought_ * smin(fwater_, fheat_)
    tau_next = tau + smax(theta - parameters.theta_base, 0.0)
    I50B_next = I50B + smax(parameters.Iwater * (1 - fwater_), parameters.Iheat * (1 - fheat_))

    # Return a new state object that holds the new values and history
    return SIMPLE_state(mB_next, tau_next, I50B_next)
end

"""
    get_yield(parameters::SIMPLE_parameters, state::SIMPLE_state)

Compute the final yield.

# Arguments
- `parameters`: SIMPLE parameters
- `state`: current SIMPLE state

# Returns
- yield
"""
function get_yield(parameters::SIMPLE_parameters, state::SIMPLE_state)
    yield = state.mB * parameters.HI
    return yield
end

"""
    f_theta(theta, x, u)

Wrapper function that simulates the model one step and takes vectors as inputs, which is useful for vectorized operations, e.g., in the particle filter.
The harvest index is not included in the parameter vector as it is not relevant for the dynamics.
A constant high CO₂ concentration of 700 ppm is assumed reducing the number of inputs to three.

# Arguments
- `theta`: vector of model parameters; theta corresponds to [tau_sum; Ia] ; theta_base; theta_opt; RUE; Iheat; Iwater; theta_heat; theta_ext; Sco2; Swater; Rmax]
- `x`: state vector; x[:,1] corresponds to [mB; tau; I50B]
- `u`: input vector; u[:,1] corresponds to [theta; D; R]

# Returns
- state vector at the next time step
"""
function f_theta end

function f_theta(theta::AbstractVector{<:AbstractFloat},
    x::Union{AbstractVector{<:AbstractFloat},AbstractMatrix{<:AbstractFloat}},
    u::Union{AbstractVector{<:AbstractFloat},AbstractMatrix{<:AbstractFloat}})

    parameters = SIMPLE_parameters(theta[1], theta[2], 6.0, 26.0, 1.00 * 1e-3, 100.0, 5.0, 32.0, 45.0, 0.07, 2.5, 0.95, 0.68)

    N = size(x, 2)
    x_next = zeros(size(x))

    for i = 1:N
        # Convert x and u from vector notation to the corresponding structs.
        state = SIMPLE_state(x[1, i], x[2, i], x[3, i])
        input = SIMPLE_input(u[1, i], u[2, i], u[3, i], 700.0)

        # Update the state.
        updated_state = step(parameters, state, input)

        # Convert the updated state back to a vector.
        x_next[:, i] = [updated_state.mB; updated_state.tau; updated_state.I50B]
    end

    return x_next
end

function f_theta(theta::AbstractVector{<:AbstractFloat},
    x::Union{AbstractVector{<:JuMP.AbstractJuMPScalar},AbstractMatrix{<:JuMP.AbstractJuMPScalar}},
    u::Union{AbstractVector{<:JuMP.AbstractJuMPScalar},AbstractMatrix{<:JuMP.AbstractJuMPScalar}})

    parameters = SIMPLE_parameters(theta[1], theta[2], 6.0, 26.0, 1.00 * 1e-3, 100.0, 5.0, 32.0, 45.0, 0.07, 2.5, 0.95, 0.68)

    N = size(x, 2)
    x_next = Array{JuMP.AbstractJuMPScalar}(undef, size(x)...)

    for i = 1:N
        # Convert x and u from vector notation to the corresponding structs.
        state = SIMPLE_state(x[1, i], x[2, i], x[3, i])
        input = SIMPLE_input(u[1, i], u[2, i], u[3, i], 700.0)

        # Update the state.
        updated_state = step(parameters, state, input)

        # Convert the updated state back to a vector.
        x_next[:, i] = [updated_state.mB; updated_state.tau; updated_state.I50B]
    end

    return x_next
end

