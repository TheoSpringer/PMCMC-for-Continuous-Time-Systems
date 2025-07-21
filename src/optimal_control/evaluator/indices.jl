# This file contains functions to handle the flat decision vector z, the vector z_scenario, and the flat constraint vector h(z).
# The flat decision vector z contains the inputs U, states X of all scenarios, outputs Y of all scenarios, and (optionally) J_max.
# The flat vector z_scenario contains the inputs U, states X_k of scenario k, outputs Y_k of scenario k, and (optionally) J_max.
# The flat constraint vector h(z) contains the dynamic, scenario, input, and (optionally) epigraph constraints.
# Note that the structure of the constraint vector h(z) also defines the structure of the vector lambda, which contains the Lagrange multipliers for the constraints in h(z).
#
# z = [ 
#       u_1(t=1);
#       u_2(t=1);
#       ...; 
#       u_{n_u}(t=1);
#       u_1(t=2);
#       ...;
#       u_{n_u}(t=H);
#       x_1(t=1)^[k=1]; 
#       x_2(t=1)^[k=1]; 
#       ...; 
#       x_{n_x}(t=1)^[k=1];
#       ...;
#       x_{n_x}(t=H)^[k=1];
#       x_1(t=1)^[k=2];
#       ...;
#       x_{n_x}(t=H)^[k=K];
#       y_1(t=1)^[k=1]; 
#       y_2(t=1)^[k=1]; 
#       ...; 
#       y_{n_y}(t=1)^[k=1];
#       ...;
#       y_{n_y}(t=H)^[k=1];
#       y_1(t=1)^[k=2];
#       ...;
#       y_{n_y}(t=H)^[k=K];
#       J_max (optional)
#       ]
#
# z_scenario[k] = [ 
#       u_1(t=1);
#       u_2(t=1);
#       ...; 
#       u_{n_u}(t=1);
#       u_1(t=2);
#       ...;
#       u_{n_u}(t=H);
#       x_1(t=1)^[k]; 
#       x_2(t=1)^[k]; 
#       ...; 
#       x_{n_x}(t=1)^[k];
#       y_1(t=1)^[k]; 
#       y_2(t=1)^[k]; 
#       ...; 
#       y_{n_y}(t=1)^[k];
#       ...;
#       y_{n_y}(t=H)^[k];
#       J_max (optional)
#       ]
#
# h(z) = [  
#       f^[k=1](x_1^[k=1],u_1) + w_1^[k=1] - x_2^[k=1];
#       f^[k=1](x_2^[k=1],u_2) + w_2^[k=1] - x_3^[k=1];
#       ...;
#       f^[k=K](x_{H-1}^[k=K],u_{H-1}) + w_{H-1}^[k=K] - x_H^[k=K];
#       g^[k=1](x_1^[k=1],u_1) + v_1^[k=1] - y_1^[k=1];
#       g^[k=1](x_2^[k=1],u_2) + v_2^[k=1] - y_2^[k=1];
#       ...;
#       g^[k=K](x_H^[k=K],u_H) + v_H^[k=K] - y_H^[k=K];
#       h_scenario^[k=1](U, X^[k=1], Y_1^[k=1]);
#       h_scenario^[k=2](U, X^[k=2], Y_1^[k=2]);
#       ...;
#       h_scenario^[k=K](U, X^[k=K], Y_1^[k=K]);
#       h_u(U);
#       J(U, X^[k=1], Y_1^[k=1]) - J_max; (optional)
#       J(U, X^[k=2], Y_1^[k=2]) - J_max; (optional)
#       ...;
#       J^[k=K](U, X^[k=K], Y_1^[k=K]) - J_max (optional)
#       ]

# This function returns the index ranges for the control inputs, the states of scenarios 1:K, the outputs of scenarios 1:K, and the maximum cost J_max inside the flat decision vector z.
function z_indices(dimensions::OCPDimensions)
    # Indices for the inputs.
    indices_U = 1:(dimensions.n_u*dimensions.H)

    # Indices for the states of all scenarios.
    indices_X = Vector{UnitRange{Int}}(undef, dimensions.K)
    first_X = last(indices_U) + 1
    for k in 1:dimensions.K
        indices_X[k] = first_X+(k-1)*dimensions.n_x*dimensions.H:first_X+k*dimensions.n_x*dimensions.H-1
    end

    # Indices for the outputs of all scenarios.
    indices_Y = Vector{UnitRange{Int}}(undef, dimensions.K)
    first_Y = first_X + dimensions.K * dimensions.n_x * dimensions.H
    for k in 1:dimensions.K
        indices_Y[k] = first_Y+(k-1)*dimensions.n_y*dimensions.H:first_Y+k*dimensions.n_y*dimensions.H-1
    end

    # Index for the maximum cost J_max (optional).
    if dimensions.J_u
        # Cost is J(u) and there is no J_max.
        index_J_max = nothing
    else
        # Epigraph notation is used and J_max is a decision variable.
        index_J_max = first_Y+dimensions.K*dimensions.n_y*dimensions.H:first_Y+dimensions.K*dimensions.n_y*dimensions.H
    end

    return indices_U, indices_X, indices_Y, index_J_max
end

# This function returns the index ranges for the dynamic constraints for the states, the dynamic constraints for the outputs, the scenario constraints h_scenario, the input constraints h_u, and (optionally) the epigraph constraints, inside the flat constraint vector h(z).
function h_indices(dimensions::OCPDimensions)
    # Indices for the dynamic constraints for the states (e.g., f^[k=1](x_1^[k=1],u_1) + w_1^[k=1] - x_2^[k=1] == 0)
    indices_h_dynamics_x = Vector{UnitRange{Int}}(undef, dimensions.K)
    first_dynamic_constraint_x = 1
    for k in 1:dimensions.K
        indices_h_dynamics_x[k] = first_dynamic_constraint_x+(k-1)*dimensions.n_x*(dimensions.H-1):first_dynamic_constraint_x+k*dimensions.n_x*(dimensions.H-1)-1
    end

    # Indices for the dynamic constraints for the outputs (e.g., g^[k=1](x_1^[k=1],u_1) + v_1^[k=1] - y_1^[k=1] == 0)
    indices_h_dynamics_y = Vector{UnitRange{Int}}(undef, dimensions.K)
    first_dynamic_constraint_y = first_dynamic_constraint_x + dimensions.K * dimensions.n_x * (dimensions.H - 1)
    for k in 1:dimensions.K
        indices_h_dynamics_y[k] = first_dynamic_constraint_y+(k-1)*dimensions.n_y*dimensions.H:first_dynamic_constraint_y+k*dimensions.n_y*dimensions.H-1

    end

    # Indices for the scenario constraints (e.g., h_scenario^[k=1](U, X^[k=1], Y_1^[k=1]) <= 0)
    indices_h_scenario = Vector{UnitRange{Int}}(undef, dimensions.K)
    first_h_scenario = first_dynamic_constraint_y + dimensions.K * dimensions.n_y * dimensions.H
    for k in 1:dimensions.K
        indices_h_scenario[k] = first_h_scenario+(k-1)*dimensions.n_h_scenario:first_h_scenario+k*dimensions.n_h_scenario-1
    end

    # Indices for the input constraints h_u(U) <= 0
    first_h_u = first_h_scenario + dimensions.K * dimensions.n_h_scenario
    indices_h_u = first_h_u:first_h_u+dimensions.n_h_u-1

    # Indices for the epigraph constraints (e.g., J(U, X^[k=1], Y_1^[k=1]) - J_max <= 0)
    indices_h_J_max = Vector{UnitRange{Int}}(undef, dimensions.K)
    current_h_J_max = first_h_u + dimensions.n_h_u
    for k in 1:dimensions.K
        if dimensions.J_u
            # Pass empty range if J_u is true
            indices_h_J_max[k] = nothing
        else
            indices_h_J_max[k] = current_h_J_max:current_h_J_max
        end
        current_h_J_max += 1
    end
    return indices_h_dynamics_x, indices_h_dynamics_y, indices_h_scenario, indices_h_u, indices_h_J_max
end

function get_indices(dimensions::OCPDimensions)
    indices_U, indices_X, indices_Y, index_J_max = z_indices(dimensions)
    indices_h_dynamics_x, indices_h_dynamics_y, indices_h_scenario, indices_h_u, indices_h_J_max = h_indices(dimensions)
    return OCPIndices(indices_U, indices_X, indices_Y, index_J_max, indices_h_dynamics_x, indices_h_dynamics_y, indices_h_scenario, indices_h_u, indices_h_J_max)
end

# This function returns the flat decision vector z corresponding to the inputs U, states X, outputs Y, and (optionally) J_max.
function pack_z(U::AbstractMatrix, X::AbstractArray, Y::AbstractArray, indices::OCPIndices, dimensions::OCPDimensions; J_max::Union{Nothing,AbstractFloat}=nothing)
    z = Vector{eltype(U)}(undef, dimensions.n_z)
    z[indices.U] .= vec(U)
    for k in 1:dimensions.K
        z[indices.X[k]] .= vec(X[:, :, k])
        z[indices.Y[k]] .= vec(Y[:, :, k])
    end
    if !dimensions.J_u
        if J_max === nothing
            z[indices.J_max] .= NaN
        else
            z[indices.J_max] .= J_max
        end
    end
    return z
end

# This function returns the inputs U, states X, outputs Y, and (optionally) J_max from the flat decision vector z.
function unpack_z(z::AbstractVector, indices::OCPIndices, dimensions::OCPDimensions)
    U = reshape((z[indices.U]), dimensions.n_u, dimensions.H)
    X = Array{eltype(z)}(undef, dimensions.n_x, dimensions.H, dimensions.K)
    Y = Array{eltype(z)}(undef, dimensions.n_y, dimensions.H, dimensions.K)
    for k in 1:dimensions.K
        X[:, :, k] .= reshape((z[indices.X[k]]), dimensions.n_x, dimensions.H)
        Y[:, :, k] .= reshape((z[indices.Y[k]]), dimensions.n_y, dimensions.H)
    end
    if !dimensions.J_u
        J_max = z[indices.J_max]
        return U, X, Y, J_max
    else
        return U, X, Y
    end
end

# This function returns the inputs U, states X_k for scenario k, outputs Y_k for scenario k, and (optionally) J_max from the flat decision vector z.
function unpack_z_k(z::AbstractVector, k::Int, indices::OCPIndices, dimensions::OCPDimensions)
    U = @views reshape(z[indices.U], dimensions.n_u, dimensions.H)
    X_k = @views reshape(z[indices.X[k]], dimensions.n_x, dimensions.H)
    Y_k = @views reshape(z[indices.Y[k]], dimensions.n_y, dimensions.H)
    if !dimensions.J_u
        J_max = z[indices.J_max]
        return U, X_k, Y_k, J_max
    else
        return U, X_k, Y_k
    end
end

# Returns the control inputs U_vec (U_vec = vec(U)) from the flat decision vector z.
function view_U_vec(z::AbstractVector, indices::OCPIndices)
    return @views z[indices.U]
end

# Returns the vector z_scenario for scenario k from the flat decision vector z.
# The vector z_scenario contains the inputs U, states X_k, outputs Y_k, and (optionally) J_max for scenario k.
function view_z_scenario(z::AbstractVector, k::Int, indices::OCPIndices, dimensions::OCPDimensions)
    U_vec = @views z[indices.U]
    X_k_vec = @views z[indices.X[k]]
    Y_k_vec = @views z[indices.Y[k]]
    if !dimensions.J_u
        J_max = @views z[indices.J_max]
        z_scenario = [U_vec; X_k_vec; Y_k_vec; J_max]
    else
        z_scenario = [U_vec; X_k_vec; Y_k_vec]
    end
    return z_scenario
end

# This function unpacks the inputs U, states X_k, outputs Y_k, and (optionally) J_max from the flat decision vector z_scenario.
function unpack_z_scenario(z_scenario::AbstractVector, dimensions::OCPDimensions)
    U = @views reshape(z_scenario[1:dimensions.n_u*dimensions.H], dimensions.n_u, dimensions.H)
    X_k = @views reshape(z_scenario[dimensions.n_u*dimensions.H+1:dimensions.n_u*dimensions.H+dimensions.n_x*dimensions.H], dimensions.n_x, dimensions.H)
    Y_k = @views reshape(z_scenario[dimensions.n_u*dimensions.H+dimensions.n_x*dimensions.H+1:dimensions.n_u*dimensions.H+dimensions.n_x*dimensions.H+dimensions.n_y*dimensions.H], dimensions.n_y, dimensions.H)
    if !dimensions.J_u
        J_max = @views z_scenario[dimensions.n_u*dimensions.H+dimensions.n_x*dimensions.H+dimensions.n_y*dimensions.H+1]
        return U, X_k, Y_k, J_max
    else
        return U, X_k, Y_k
    end
end

# The following function translates indices with respect to the vector z_scenario to indices with respect to the global decision vector z.
# The inputs offset_X and offset_Y are the offsets of the first entries of the state and output blocks of the considered scenario in the global decision vector z.
function translate_local_index_to_global(local_index::Int, offset_X::Int, offset_Y::Int, dimensions::OCPDimensions, indices::OCPIndices)
    if local_index <= dimensions.n_u * dimensions.H
        # Entry belongs to the input block.
        global_index = indices.U[local_index]
    elseif local_index <= dimensions.n_u * dimensions.H + dimensions.n_x * dimensions.H
        # Entry belongs to the state block.
        local_indices_X = local_index - dimensions.n_u * dimensions.H
        global_index = offset_X + local_indices_X
    elseif local_index <= dimensions.n_u * dimensions.H + dimensions.n_x * dimensions.H + dimensions.n_y * dimensions.H
        # Entry belongs to the output block.
        local_indices_Y = local_index - dimensions.n_u * dimensions.H - dimensions.n_x * dimensions.H
        global_index = offset_Y + local_indices_Y
    else
        # Entry belongs to the J_max constraint.
        global_index = first(indices.J_max)
    end
    return global_index
end