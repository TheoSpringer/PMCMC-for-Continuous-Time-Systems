# This file contains functions to handle the flat decision vector z containing the inputs U, states X of all scenarios, outputs Y of all scenarios, and (optionally) J_max,
# the flat vector z_scenario containing the inputs U, states X_k for scenario k, outputs Y_k for scenario k, and (optionally) J_max,
# and the flat constraint vector h(z) containing the dynamic, scenario, control, and (optionally) epigraph constraints.
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
#       y_{n_y}(t=H)^[k=K]
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
#       y_{n_y}(t=H)^[k]
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
function z_indices(n_u::Int, n_x::Int, n_y::Int, H::Int, K::Int, J_u::Bool)
    # Indices for the inputs.
    indices_U = 1:(n_u*H)

    # Indices for the states of all scenarios.
    indices_X = Vector{UnitRange{Int}}(undef, K)
    first_X = last(indices_U) + 1
    for k in 1:K
        indices_X[k] = first_X+(k-1)*n_x*H:first_X+k*n_x*H-1
    end

    # Indices for the outputs of all scenarios.
    indices_Y = Vector{UnitRange{Int}}(undef, K)
    first_Y = first_X + K * n_x * H
    for k in 1:K
        indices_Y[k] = first_Y+(k-1)*n_y*H:first_Y+k*n_y*H-1
    end

    # Index for the maximum cost J_max (optional).
    if J_u
        # Cost is J(u) and there is no J_max.
        index_J_max = 1:0 # empty range
    else
        # Epigraph notation is used and J_max is a decision variable.
        index_J_max = first_Y+K*n_y*H:first_Y+K*n_y*H
    end

    return indices_U, indices_X, indices_Y, index_J_max
end

# This function returns the flat decision vector z corresponding to the inputs U, states X, outputs Y, and (optionally) J_max.
function pack_z(U::AbstractMatrix, X::AbstractArray, Y::AbstractArray, indices_U::UnitRange{Int}, indices_X::Vector{UnitRange{Int}}, indices_Y::Vector{UnitRange{Int}}, index_J_max::UnitRange{Int}, K::Int, n_z::Int, J_u::Bool; J_max::Union{Nothing,AbstractFloat}=nothing)
    z = Vector{eltype(U)}(undef, n_z)
    z[indices_U] .= vec(U)
    for k in 1:K
        z[indices_X[k]] .= vec(X[:, :, k])
        z[indices_Y[k]] .= vec(Y[:, :, k])
    end
    if !J_u
        if J_max === nothing
            z[index_J_max] .= NaN
        else
            z[index_J_max] .= J_max
        end
    end
    return z
end

# This function returns the inputs U, states X, outputs Y, and (optionally) J_max from the flat decision vector z.
function unpack_z(z::AbstractVector, indices_U::UnitRange{Int}, indices_X::Vector{UnitRange{Int}}, indices_Y::Vector{UnitRange{Int}}, index_J_max::UnitRange{Int}, K::Int, n_u::Int, n_x::Int, n_y::Int, H::Int, J_u::Bool)
    U = reshape((z[indices_U]), n_u, H)
    X = Array{eltype(z)}(undef, n_x, H, K)
    Y = Array{eltype(z)}(undef, n_y, H, K)
    for k in 1:K
        X[:, :, k] .= reshape((z[indices_X[k]]), n_x, H)
        Y[:, :, k] .= reshape((z[indices_Y[k]]), n_y, H)
    end
    if !J_u
        J_max = z[index_J_max]
        return U, X, Y, J_max
    else
        return U, X, Y
    end
end

# This function returns the inputs U, states X_k, outputs Y_k, and (optionally) J_max for scenario k from the flat decision vector z.
function unpack_z_k(z::AbstractVector, k::Int, indices_U::UnitRange{Int}, indices_X::Vector{UnitRange{Int}}, indices_Y::Vector{UnitRange{Int}}, index_J_max::UnitRange{Int}, n_u::Int, n_x::Int, n_y::Int, H::Int, J_u::Bool)
    U = @views reshape(z[indices_U], n_u, H)
    X_k = @views reshape(z[indices_X[k]], n_x, H)
    Y_k = @views reshape(z[indices_Y[k]], n_y, H)
    if !J_u
        J_max = z[index_J_max]
        return U, X_k, Y_k, J_max
    else
        return U, X_k, Y_k
    end
end

# Returns the control inputs U_vec from the flat decision vector z containing all decision variables.
function view_U_vec(z::AbstractVector, indices_U::UnitRange{Int})
    return @views z[indices_U]
end

# Returns the decision vector z_scenario that contains the inputs U, states X_k, outputs Y_k, and (optionally) J_max for scenario k from the flat decision vector z containing all decision variables.
function view_z_scenario(z::AbstractVector, k::Int, indices_U::UnitRange{Int}, indices_X::Vector{UnitRange{Int}}, indices_Y::Vector{UnitRange{Int}}, index_J_max::UnitRange{Int}, J_u::Bool)
    U_vec = @views z[indices_U]
    X_k_vec = @views z[indices_X[k]]
    Y_k_vec = @views z[indices_Y[k]]
    if !J_u
        J_max = @views z[index_J_max]
        z_scenario = [U_vec; X_k_vec; Y_k_vec; J_max]
    else
        z_scenario = [U_vec; X_k_vec; Y_k_vec]
    end
    return z_scenario
end

# This function unpacks the inputs U, states X_k, outputs Y_k, and (optionally) J_max from the flat decision vector z_scenario that contains the decision variables for a single scenario.
function unpack_z_scenario(z_scenario::AbstractVector, n_u::Int, n_x::Int, n_y::Int, H::Int, J_u::Bool)
    U = @views reshape(z_scenario[1:n_u*H], n_u, H)
    X_k = @views reshape(z_scenario[n_u*H+1:n_u*H+n_x*H], n_x, H)
    Y_k = @views reshape(z_scenario[n_u*H+n_x*H+1:n_u*H+n_x*H+n_y*H], n_y, H)
    if !J_u
        J_max = @views z_scenario[n_u*H+n_x*H+n_y*H+1]
        return U, X_k, Y_k, J_max
    else
        return U, X_k, Y_k
    end
end

# This function returns the index ranges for dynamic constraints, the scenario constraints h_scenario, and the input constraints h_u inside the flat constraint vector h(z).
function h_indices(n_x::Int, n_y::Int, H::Int, K::Int, n_h_scenario::Int, n_h_u::Int, J_u::Bool)
    # Indices for the dynamic constraints for the states (e.g., f^[k=1](x_1^[k=1],u_1) + w_1^[k=1] - x_2^[k=1] == 0)
    indices_h_dynamics_x = Vector{UnitRange{Int}}(undef, K)
    first_dynamic_constraint_x = 1
    for k in 1:K
        indices_h_dynamics_x[k] = first_dynamic_constraint_x+(k-1)*n_x*(H-1):first_dynamic_constraint_x+k*n_x*(H-1)-1
    end

    # Indices for the dynamic constraints for the outputs (e.g., g^[k=1](x_1^[k=1],u_1) + v_1^[k=1] - y_1^[k=1] == 0)
    indices_h_dynamics_y = Vector{UnitRange{Int}}(undef, K)
    first_dynamic_constraint_y = first_dynamic_constraint_x + K * n_x * (H - 1)
    for k in 1:K
        indices_h_dynamics_y[k] = first_dynamic_constraint_y+(k-1)*n_y*H:first_dynamic_constraint_y+k*n_y*H-1

    end

    # Indices for the scenario constraints (e.g., h_scenario^[k=1](U, X^[k=1], Y_1^[k=1]) <= 0)
    indices_h_scenario = Vector{UnitRange{Int}}(undef, K)
    first_h_scenario = first_dynamic_constraint_y + K * n_y * H
    for k in 1:K
        indices_h_scenario[k] = first_h_scenario+(k-1)*n_h_scenario:first_h_scenario+k*n_h_scenario-1
    end

    # Indices for the input constraints h_u(U) <= 0
    first_h_u = first_h_scenario + K * n_h_scenario
    indices_h_u = first_h_u:first_h_u+n_h_u-1

    # Indices for the J^[k] - J_max <= 0 constraint (e.g., J(U, X^[k=1], Y_1^[k=1]) - J_max <= 0)
    indices_h_J_max = Vector{UnitRange{Int}}(undef, K)
    current_h_J_max = first_h_u + n_h_u
    for k in 1:K
        if J_u
            # Pass empty range if J_u is true
            indices_h_J_max[k] = 1:0 # empty range
        else
            indices_h_J_max[k] = current_h_J_max:current_h_J_max
        end
        current_h_J_max += 1
    end
    return indices_h_dynamics_x, indices_h_dynamics_y, indices_h_scenario, indices_h_u, indices_h_J_max
end

# This function expands the local pattern of the Jacobian of a constraint with respect to a single scenario (i.e., the Jacobian with respect to z_scenario) to the global index space (i.e., the Jacobian of the constraints for all scenarios with respect to z).
# The corresponding indices of nonzero elements are added to the sparsity_global_Jacobian_rows and sparsity_global_Jacobian_columns vectors.
# The input indices_h_global[k] contains the indices of the constraint corresponding to scenario k in the global constraint vector h(z).
# The function returns the the ranges of the constraint for each scenario in the vector containing the non-zero entries of the global Jacobian.
function expand_local_Jacobian_sparsity_pattern!(sparsity_global_Jacobian_rows::AbstractVector{<:Integer}, sparsity_global_Jacobian_columns::AbstractVector{<:Integer}, sparsity_local_Jacobian_rows::Vector{Int}, sparsity_local_Jacobian_columns::Vector{Int}, indices_h_global::Vector{UnitRange{Int}}, indices_U::UnitRange{Int}, indices_X::Vector{UnitRange{Int}}, indices_Y::Vector{UnitRange{Int}}, index_J_max::UnitRange{Int}, n_u::Int, n_x::Int, n_y::Int, H::Int, K::Int)
    nzvals_Jacobian_ranges = Vector{UnitRange{Int}}(undef, K)
    for k in 1:K
        row_offset = first(indices_h_global[k]) - 1 # offset of the first row of the constraint in the global constraint vector h(z)
        column_offset_X = first(indices_X[k]) - 1 # offset of the first column of the state x_{1:H}^{[k]} of scenario k in the global variable vector z
        column_offset_Y = first(indices_Y[k]) - 1 # offset of the first column of the output y_{1:H}^{[k]} of scenario k in the global variable vector z
        start = length(sparsity_global_Jacobian_rows) + 1 # start index of the non-zero entries of the Jacobian of scenario k in the vector containing the non-zero entries of the global Jacobian
        for (r, c) in zip(sparsity_local_Jacobian_rows, sparsity_local_Jacobian_columns)
            global_row = row_offset + r
            if c <= n_u * H
                # Entry belongs to the input block.
                global_column = indices_U[c]
            elseif c <= n_u * H + n_x * H
                # Entry belongs to the state block.
                local_indices_X = c - n_u * H
                global_column = column_offset_X + local_indices_X
            elseif c <= n_u * H + n_x * H + n_y * H
                # Entry belongs to the output block.
                local_indices_Y = c - n_u * H - n_x * H
                global_column = column_offset_Y + local_indices_Y
            else
                # Entry belongs to the J_max constraint.
                global_column = first(index_J_max)
            end
            # Add the global row and column indices.
            push!(sparsity_global_Jacobian_rows, global_row)
            push!(sparsity_global_Jacobian_columns, global_column)
        end
        # Get the range of non-zero entries of the Jacobian for scenario k in the vector containing the non-zero entries of the global Jacobian.
        stop = length(sparsity_global_Jacobian_rows)
        nzvals_Jacobian_ranges[k] = start:stop
    end
    return nzvals_Jacobian_ranges, sparsity_global_Jacobian_rows, sparsity_global_Jacobian_columns
end