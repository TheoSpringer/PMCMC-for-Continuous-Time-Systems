# This file contains the implementation of a custom MathOptInterface evaluator for a scenario-based optimal control problem.
# By exploiting the specific structure of the problem and the sparsity pattern, the evaluator can efficiently compute the objective, the objective gradient, the constraints and the constraint Jacobian.
# The evaluator utilizes thread parallelism to speed up the evaluation of the constraints and their Jacobian.
# To profit from the implementation ensure that multiple threads are available by setting the JULIA_NUM_THREADS environment variable.

using LinearAlgebra
using Symbolics
using SparseArrays
using ForwardDiff
using SparseDiffTools
using JuMP
using Ipopt
using MathOptInterface
const MOI = MathOptInterface

# This function returns the index ranges for the control inputs, the states of scenarios 1:K, the outputs of scenarios 1:K, and the maximum cost J_max inside the flat decision vector z.
# Decision vector layout: z = [vec(U); vec(X^[1]) … vec(X^[K]); vec(Y^[1]) … vec(Y^[K]); J_max (optional)]
function z_indices(n_u::Int, n_x::Int, n_y::Int, H::Int, K::Int, J_u::Bool)
    # Inputs
    indices_U = 1:(n_u*H)

    # States for all scenarios
    indices_X = Vector{UnitRange{Int}}(undef, K)
    first_X = last(indices_U) + 1
    for k in 1:K
        indices_X[k] = first_X+(k-1)*n_x*H:first_X+k*n_x*H-1
    end

    # Outputs for all scenarios
    indices_Y = Vector{UnitRange{Int}}(undef, K)
    first_Y = first_X + K * n_x * H
    for k in 1:K
        indices_Y[k] = first_Y+(k-1)*n_y*H:first_Y+k*n_y*H-1
    end

    # Maximum cost J_max (optional)
    if J_u
        # Cost is J(u) and there is no J_max.
        index_J_max = 1:0 # empty range
    else
        index_J_max = first_Y+K*n_y*H:first_Y+K*n_y*H
    end

    return indices_U, indices_X, indices_Y, index_J_max
end

# This function returns the flat decision vector z from the inputs U, states X, outputs Y, and (optionally) J_max.
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

# Returns the control inputs U_vec from the flat decision vector z containing all decision variables.
function view_U_vec(z::AbstractVector, indices_U::UnitRange{Int})
    return @views z[indices_U]
end

# This function returns the index ranges for dynamic constraints, the scenario constraints h_scenario, and the input constraints h_u inside the flat constraint vector h(z).
# h(z) = [  f^[k=1](x_1^[k=1],u_1) + w_1^[k=1] - x_2^[k=1];
#           f^[k=1](x_2^[k=1],u_2) + w_2^[k=1] - x_3^[k=1];
#           ...;
#           f^[k=K](x_{H-1}^[k=K],u_{H-1}) + w_{H-1}^[k=K] - x_H^[k=K];
#           g^[k=1](x_1^[k=1],u_1) + v_1^[k=1] - y_1^[k=1];
#           g^[k=1](x_2^[k=1],u_2) + v_2^[k=1] - y_2^[k=1];
#           ...;
#           g^[k=K](x_H^[k=K],u_H) + v_H^[k=K] - y_H^[k=K];
#           h_scenario^[k=1](U, X^[k=1], Y_1^[k=1]);
#           h_scenario^[k=2](U, X^[k=2], Y_1^[k=2]);
#           ...;
#           h_scenario^[k=K](U, X^[k=K], Y_1^[k=K]);
#           h_u(U);
#           J(U, X^[k=1], Y_1^[k=1]) - J_max; (optional)
#           J(U, X^[k=2], Y_1^[k=2]) - J_max; (optional)
#           ...;
#           J^[k=K](U, X^[k=K], Y_1^[k=K]) - J_max (optional)] 
#
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

# Evaluator struct.
struct PMMH_OCP_Evaluator <: MOI.AbstractNLPEvaluator
    # Dimensions
    K::Int
    H::Int
    n_u::Int
    n_x::Int
    n_y::Int
    n_z::Int # total number of decision add_variables
    n_h::Int # total number of constraints

    # Data
    PMMH_samples::Vector{PMMH_sample}
    V::Array{Float64,3}
    W::Array{Float64,3}
    X_t::Array{Float64,2} # initial states for the scenarios

    # Functions
    h_dynamics_x!::Function # dynamics constraints for the states
    h_dynamics_y!::Function # dynamics constraints for the outputs
    h_scenario!::Function # scenario constraints
    h_u!::Function # input constraints
    h_J_max!::Function # J - J_max constraint (optional)
    eval_J_u::Function # evaluates J(u)
    J_u::Bool

    # Index ranges 
    indices_U::UnitRange{Int}
    indices_X::Vector{UnitRange{Int}}
    indices_Y::Vector{UnitRange{Int}}
    index_J_max::UnitRange{Int}

    # Index ranges for the constraints
    indices_h_dynamics_x::Vector{UnitRange{Int}}
    indices_h_dynamics_y::Vector{UnitRange{Int}}
    indices_h_scenario::Vector{UnitRange{Int}}
    indices_h_u::UnitRange{Int}
    indices_h_J_max::Vector{UnitRange{Int}}

    # Bounds for the decision vector z
    z_sets::Vector{MOI.AbstractScalarSet}

    # Bounds for the constraint vector h(z)
    h_bounds::Vector{MOI.NLPBoundsPair}

    # Global sparsity pattern
    Jac_pattern_h::Vector{Tuple{Int,Int}}

    # Local sparsity pattern and coloring for the dynamics constraints of one scenario
    Jac_pattern_h_dynamics_x::SparseMatrixCSC{Bool,Int}
    colors_h_dynamics_x::Vector{Int}
    nzrange_h_dynamics_x::Vector{UnitRange{Int}}

    Jac_pattern_h_dynamics_y::SparseMatrixCSC{Bool,Int}
    colors_h_dynamics_y::Vector{Int}
    nzrange_h_dynamics_y::Vector{UnitRange{Int}}

    # Local sparsity pattern and coloring for the scenario constraints h_scenario of one scenario
    Jac_pattern_h_scenario::SparseMatrixCSC{Bool,Int}
    colors_h_scenario::Vector{Int}
    nzrange_h_scenario::Vector{UnitRange{Int}}

    # Sparsity pattern and coloring for h_u
    Jac_pattern_h_u::SparseMatrixCSC{Bool,Int}
    colors_h_u::Vector{Int}
    nzrange_h_u::UnitRange{Int}

    # Local sparsity pattern for the J - J_max (epigraph) constraint
    Jac_pattern_h_J_max::SparseMatrixCSC{Bool,Int}
    colors_h_J_max::Vector{Int}
    nzrange_h_J_max::Vector{UnitRange{Int}}
end

# Constructor for PMMH_OCP_Evaluator. The sparsity pattern is built once for a single scenario and then replicated for all K scenarios.
function PMMH_OCP_Evaluator(PMMH_samples::Vector{PMMH_sample}, V::Array{Float64}, W::Array{Float64}, X_t::Array{Float64}, f_theta::Function, g_theta::Function, J::Function, J_u::Bool, h_scenario::Function, h_u::Function, n_u::Int, n_x::Int, n_y::Int, H::Int)
    # Check if multithreading is enabled.
    n_threads = Threads.nthreads()

    @info "Evaluator running with $n_threads Julia thread$(n_threads == 1 ? "" : "s")."

    if n_threads == 1
        @warn "Multithreading is disabled (JULIA_NUM_THREADS = 1).\n" *
              "Jacobian computations and other parallel loops will run serially. " *
              "Enable multithreading for better performance."
    end

    # Get the number of decision variables.
    K = length(PMMH_samples)
    if !J_u
        n_z = n_u * H + n_x * H + n_y * H + 1
    else
        n_z = n_u * H + n_x * H + n_y * H
    end

    # Get the index ranges in the flat decision vector z corresponding to the inputs, states, outputs, and (optionally) J_max.
    indices_U, indices_X, indices_Y, index_J_max = z_indices(n_u, n_x, n_y, H, K, J_u)

    # Get bounds for the decision vector z.
    z_sets = Vector{MOI.AbstractScalarSet}(undef, n_z)
    for i in indices_U
        z_sets[i] = MOI.Interval(-Inf, Inf)
    end
    for k in 1:K
        for i in 1:length(indices_X[k])
            if i <= n_x
                # Initial state x_1^[k] is fixed.
                z_sets[indices_X[k][i]] = MOI.EqualTo(X_t[i, k])
            else
                z_sets[indices_X[k][i]] = MOI.Interval(-Inf, Inf)
            end
        end
        for i in indices_Y[k]
            z_sets[i] = MOI.Interval(-Inf, Inf)
        end
    end
    if !J_u
        for i in index_J_max
            z_sets[i] = MOI.Interval(-Inf, Inf)
        end
    end

    # Get the size of the constraints.
    h_test = h_scenario(zeros(n_u, H), zeros(n_x, H), zeros(n_y, H))
    if isa(h_test, AbstractArray)
        n_h_scenario = length(vec(h_test))
    else
        n_h_scenario = 1
    end

    h_u_test = h_u(zeros(n_u, H))
    if isa(h_u_test, AbstractArray)
        n_h_u = length(vec(h_u_test))
    else
        n_h_u = 1
    end

    if !J_u
        n_h = K * n_x * (H - 1) + K * n_y * H + K * n_h_scenario + n_h_u + K
    else
        n_h = K * n_x * (H - 1) + K * n_y * H + K * n_h_scenario + n_h_u
    end

    # Get the index ranges in the flat constraint vector h(z) corresponding to the dynamic constraints, scenario constraints h_scenario, input constraints h_u, and the J - J_max constraint.
    indices_h_dynamics_x, indices_h_dynamics_y, indices_h_scenario, indices_h_u, indices_h_J_max = h_indices(n_x, n_y, H, K, n_h_scenario, n_h_u, J_u)

    # Get bounds for the constraint vector h(z).
    h_bounds = Vector{MOI.NLPBoundsPair}(undef, n_h)
    for k in 1:K
        # Dynamic (equality) constraints for the states.
        for i in indices_h_dynamics_x[k]
            h_bounds[i] = MOI.NLPBoundsPair(0.0, 0.0)
        end

        # Dynamic (equality) constraints for the outputs.
        for i in indices_h_dynamics_y[k]
            h_bounds[i] = MOI.NLPBoundsPair(0.0, 0.0)
        end

        # Scenario (inequality) constraints.
        for i in indices_h_scenario[k]
            h_bounds[i] = MOI.NLPBoundsPair(-Inf, 0.0)
        end

        # J - J_max (inequality) constraints.
        if !J_u
            for i in indices_h_J_max[k]
                h_bounds[i] = MOI.NLPBoundsPair(-Inf, 0.0)
            end
        end
    end

    # Input (inequality) constraints.
    for i in indices_h_u
        h_bounds[i] = MOI.NLPBoundsPair(-Inf, 0.0)
    end

    # Determine the sparsity pattern for the Jacobian of the dynamics constraints.
    # The following vectors determine in which region the sparsity pattern is evaluated.
    # Without input dependent branches (e.g., min, max, if) the expression tree is fixed and the sparsity pattern does not depend on the actual values of these vectors.
    z_scenario = zeros(n_z)
    h_dynamics_x_loc = zeros(n_x * (H - 1))
    h_dynamics_y_loc = zeros(n_y * H)

    # Helper function that evaluates the dynamics constraints over the whole horizon H for one scenario.
    function h_dynamics_x!(h_dyn_x::AbstractVector, U::AbstractMatrix, X_k::AbstractMatrix, theta::AbstractArray, V_k::AbstractMatrix)
        for t in 1:H-1
            h_dyn_x[(t-1)*n_x+1:t*n_x] .= f_theta(theta, X_k[:, t], U[:, t]) .+ V_k[:, t] .- X_k[:, t+1]
        end
        return h_dyn_x
    end

    # Helper function that returns the constraint vector containing the dynamics constraints over the whole horizon H for one scenario with a vector valued input z = [vec(U); vec(X^[k]); vec(Y^[k])].
    function h_dynamics_x!(h_dyn_x::AbstractVector, z_scenario::AbstractVector, theta::AbstractArray, V_k::AbstractMatrix)
        U, X_k = unpack_z_scenario(z_scenario, n_u, n_x, n_y, H, J_u)[1:2]
        return h_dynamics_x!(h_dyn_x, U, X_k, theta, V_k)
    end

    # Helper function that evaluates the measurement constraints over the whole horizon H for one scenario.
    function h_dynamics_y!(h_dyn_y::AbstractVector, U::AbstractMatrix, X_k::AbstractMatrix, Y_k::AbstractMatrix, theta::AbstractArray, W_k::AbstractMatrix)
        for t in 1:H
            h_dyn_y[(t-1)*n_y+1:t*n_y] .= g_theta(theta, X_k[:, t], U[:, t]) .+ W_k[:, t] .- Y_k[:, t]
        end
        return h_dyn_y
    end

    function h_dynamics_y!(h_dyn_y::AbstractVector, z_scenario::AbstractVector, theta::AbstractArray, W_k::AbstractMatrix)
        U, X_k, Y_k = unpack_z_scenario(z_scenario, n_u, n_x, n_y, H, J_u)[1:3]
        return h_dynamics_y!(h_dyn_y, U, X_k, Y_k, theta, W_k)
    end

    # Evaluate the sparsity pattern of the Jacobian of the dynamics constraints for a single scenario.
    Jac_pattern_h_dynamics_x = Symbolics.jacobian_sparsity((h_dyn_x, z) -> h_dynamics_x!(h_dyn_x, z, PMMH_samples[1].theta, V[:, :, 1]), h_dynamics_x_loc, z_scenario)
    Jac_pattern_h_dynamics_y = Symbolics.jacobian_sparsity((h_dyn_y, z) -> h_dynamics_y!(h_dyn_y, z, PMMH_samples[1].theta, W[:, :, 1]), h_dynamics_y_loc, z_scenario)

    # Get matrix coloring.
    colors_h_dynamics_x = SparseDiffTools.matrix_colors(Jac_pattern_h_dynamics_x)
    colors_h_dynamics_y = SparseDiffTools.matrix_colors(Jac_pattern_h_dynamics_y)

    # Find the non-zero entries in the sparsity pattern of the dynamics constraints.
    rows_h_dynamics_x_local, columns_h_dynamics_x_local, _ = findnz(Jac_pattern_h_dynamics_x)
    rows_h_dynamics_y_local, columns_h_dynamics_y_local, _ = findnz(Jac_pattern_h_dynamics_y)

    # Initialize sparsity pattern for the constraint Jacobian.
    sparsity_Jacobian_rows = Int32[] # global row indices of non-zero entries
    sparsity_Jacobian_columns = Int32[] # global column indices of non-zero entries

    # The following vectors contain the ranges of the dynamic constraints for each scenario in the flat vector J_val that contains the non-zero elements of the constraint Jacobian; see MOI.eval_constraint_jacobian().
    nzrange_h_dynamics_x = Vector{UnitRange{Int}}(undef, K)
    nzrange_h_dynamics_y = Vector{UnitRange{Int}}(undef, K)

    # Loop over the scenarios and shift the local sparsity pattern.
    for k in 1:K
        row_offset_x = first(indices_h_dynamics_x[k]) # first row of the dynamic constraints of scenario k in the global constraint vector
        column_offset_X = first(indices_X[k]) - 1 # first column of the state x_{1:H}^{[k]} of scenario k in the global variable vector
        column_offset_Y = first(indices_Y[k]) - 1 # first column of the output y_{1:H}^{[k]} of scenario k in the global variable vector
        start = length(sparsity_Jacobian_rows) + 1
        for (r, c) in zip(rows_h_dynamics_x_local, columns_h_dynamics_x_local)
            global_row = row_offset_x + r
            if c <= n_u * H
                # Entry belongs to the input block.
                global_column = indices_U[c]
            elseif c <= n_u * H + n_x * H
                # Entry belongs to the state block.
                local_indices_X = c - n_u * H
                global_column = column_offset_X + local_indices_X
            else
                # Entry belongs to the output block.
                local_indices_Y = c - n_u * H - n_x * H
                global_column = column_offset_Y + local_indices_Y
            end

            # Add the global row and column indices to the lists.
            push!(sparsity_Jacobian_rows, global_row)
            push!(sparsity_Jacobian_columns, global_column)
        end
        stop = length(sparsity_Jacobian_rows)
        nzrange_h_dynamics_x[k] = start:stop
    end

    for k in 1:K
        row_offset_y = first(indices_h_dynamics_y[k]) # first row of the measurement constraints of scenario k in the global constraint vector
        column_offset_X = first(indices_X[k]) - 1 # first column of the state x_{1:H}^{[k]} of scenario k in the global variable vector
        column_offset_Y = first(indices_Y[k]) - 1 # first column of the output y_{1:H}^{[k]} of scenario k in the global variable vector
        start = length(sparsity_Jacobian_rows) + 1
        for (r, c) in zip(rows_h_dynamics_y_local, columns_h_dynamics_y_local)
            global_row = row_offset_y + r
            if c <= n_u * H
                # Entry belongs to the input block.
                global_column = indices_U[c]
            elseif c <= n_u * H + n_x * H
                # Entry belongs to the state block.
                local_indices_X = c - n_u * H
                global_column = column_offset_X + local_indices_X
            else
                # Entry belongs to the output block.
                local_indices_Y = c - n_u * H - n_x * H
                global_column = column_offset_Y + local_indices_Y
            end

            # Add the global row and column indices to the lists.
            push!(sparsity_Jacobian_rows, global_row)
            push!(sparsity_Jacobian_columns, global_column)
        end
        stop = length(sparsity_Jacobian_rows)
        nzrange_h_dynamics_y[k] = start:stop
    end

    # Determine the sparsity pattern for the Jacobian of h_scenario.
    # The following vectors determine in which region the sparsity pattern is evaluated.
    # Without input dependent branches (e.g., min, max, if) the expression tree is fixed and the sparsity pattern does not depend on the actual values of these vectors.
    h_scenario_loc = zeros(n_h_scenario)

    # Helper function that evaluates constraints for a single scenario.
    function h_scenario!(h::AbstractVector, U::AbstractMatrix, X_k::AbstractMatrix, Y_k::AbstractMatrix)
        h .= vec(h_scenario(U, X_k, Y_k))
        return h
    end

    function h_scenario!(h::AbstractVector, z_scenario::AbstractVector)
        U, X_k, Y_k = unpack_z_scenario(z_scenario, n_u, n_x, n_y, H, J_u)[1:3]
        return h_scenario!(h, U, X_k, Y_k)
    end

    # Evaluate the sparsity pattern of the Jacobian of h_scenario for a single scenario.
    Jac_pattern_h_scenario = Symbolics.jacobian_sparsity(h_scenario!, h_scenario_loc, z_scenario)

    # Get matrix coloring.
    colors_h_scenario = SparseDiffTools.matrix_colors(Jac_pattern_h_scenario)

    # Find the non-zero entries in the sparsity pattern of a single scenario constraint.
    rows_h_scenario_local, columns_h_scenario_local, _ = findnz(Jac_pattern_h_scenario)

    # Initialize ranges of the scenario constraints for each scenario in the flat vector J_val that contains the non-zero elements of the constraint Jacobian.
    nzrange_h_scenario = Vector{UnitRange{Int}}(undef, K)

    # Loop over the scenarios and shift the local sparsity pattern.
    for k in 1:K
        row_offset = first(indices_h_scenario[k]) # first row of h_scenario() of scenario k in the global constraint vector
        column_offset_X = first(indices_X[k]) - 1 # first column of the state x_{1:H}^{[k]} of scenario k in the global variable vector
        column_offset_Y = first(indices_Y[k]) - 1 # first column of the output y_{1:H}^{[k]} of scenario k in the global variable vector
        start = length(sparsity_Jacobian_rows) + 1
        for (r, c) in zip(rows_h_scenario_local, columns_h_scenario_local)
            global_row = row_offset + r

            if c <= n_u * H
                # Entry belongs to the input block.
                global_column = indices_U[c]
            elseif c <= n_u * H + n_x * H
                # Entry belongs to the state block.
                local_indices_X = c - n_u * H
                global_column = column_offset_X + local_indices_X

            else
                # Entry belongs to the output block.
                local_indices_Y = c - n_u * H - n_x * H
                global_column = column_offset_Y + local_indices_Y
            end

            # Add the global row and column indices to the lists.
            push!(sparsity_Jacobian_rows, global_row)
            push!(sparsity_Jacobian_columns, global_column)
        end
        stop = length(sparsity_Jacobian_rows)
        nzrange_h_scenario[k] = start:stop
    end

    # Get size of input constraints.
    n_h_u = length(h_u(zeros(n_u, H)))

    # Determine the sparsity pattern for the Jacobian of h_u.
    # The following vectors determine in which region the sparsity pattern is evaluated.
    # Without input dependent branches (e.g., min, max, if) the expression tree is fixed and the sparsity pattern does not depend on the actual values of these vectors.
    u_loc = zeros(n_u * H)
    h_u_loc = zeros(n_h_u)

    # Helper function that evaluates h_u.
    function h_u!(h::AbstractVector, U::AbstractMatrix)
        h .= vec(h_u(U))
        return h
    end

    function h_u!(h::AbstractVector, U_vec::AbstractVector)
        U = @views reshape(U_vec, n_u, H)
        return h_u!(h, U)
    end

    # Evaluate the sparsity pattern of h_u.
    Jac_pattern_h_u = Symbolics.jacobian_sparsity(h_u!, h_u_loc, u_loc)

    # Get matrix coloring.
    colors_h_u = SparseDiffTools.matrix_colors(Jac_pattern_h_u)

    # Find the non-zero entries in the sparsity pattern of h_u.
    rows_h_u_local, columns_h_u_local, _ = findnz(Jac_pattern_h_u)

    row_offset = first(indices_h_u)
    start = length(sparsity_Jacobian_rows) + 1
    for (r, c) in zip(rows_h_u_local, columns_h_u_local)
        push!(sparsity_Jacobian_rows, row_offset + r)
        push!(sparsity_Jacobian_columns, indices_U[c])
    end
    stop = length(sparsity_Jacobian_rows)
    nzrange_h_u = start:stop

    if !J_u
        # Evaluate the sparsity pattern of the Jacobian of J^[k] - J_max (constraint utilized to minimize the worst-case cost).
        # The following vectors determine in which region the sparsity pattern is evaluated.
        J_loc = zeros(1)

        function h_J_max!(h::AbstractVector, U::AbstractMatrix, X_k::AbstractMatrix, Y_k::AbstractMatrix, J_max::Union{<:Number,AbstractVector{<:Number}})
            h .= J(U, X_k, Y_k) .- J_max
            return h
        end

        function h_J_max!(h::AbstractVector, z_scenario::AbstractVector)
            U, X_k, Y_k, J_max = unpack_z_scenario(z_scenario, n_u, n_x, n_y, H, J_u)
            return h_J_max!(h, U, X_k, Y_k, J_max)
        end

        # Evaluate the sparsity pattern of the Jacobian of h_scenario for a single scenario.
        Jac_pattern_h_J_max = Symbolics.jacobian_sparsity(h_J_max!, J_loc, z_scenario)

        # Get matrix coloring.
        colors_h_J_max = SparseDiffTools.matrix_colors(Jac_pattern_h_J_max)

        # Find the non-zero entries in the sparsity pattern of a single constraint.
        _, cols_h_J_max, _ = findnz(Jac_pattern_h_J_max)

        # Initialize ranges of the scenario constraints for each scenario in the flat vector J_val that contains the non-zero elements of the constraint Jacobian.
        nzrange_h_J_max = Vector{UnitRange{Int}}(undef, K)

        # Loop over the scenarios and shift the local sparsity pattern.
        for k in 1:K
            global_row = first(indices_h_J_max[k]) # first row of h_scenario() of scenario k in the global constraint vector
            column_offset_X = first(indices_X[k]) - 1 # first column of the state x_{1:H}^{[k]} of scenario k in the global variable vector
            column_offset_Y = first(indices_Y[k]) - 1 # first column of the output y_{1:H}^{[k]} of scenario k in the global variable vector
            start = length(sparsity_Jacobian_rows) + 1
            for c in cols_h_J_max
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
                # Add the global row and column indices to the lists.
                push!(sparsity_Jacobian_rows, global_row)
                push!(sparsity_Jacobian_columns, global_column)
            end
            stop = length(sparsity_Jacobian_rows)
            nzrange_h_J_max[k] = start:stop
        end

        eval_J_u = (U_vec) -> nothing
    else
        # If J_u is true, the J_max constraint is not used.
        h_J_max! = (h, z_scenario) -> nothing
        Jac_pattern_h_J_max = SparseMatrixCSC{Bool,Int}(undef, 0, 0)
        colors_h_J_max = Int[]
        nzrange_h_J_max = 1:0

        # To compute the gradient of J(U), we need to define a vectorized version of J(U).
        function eval_J_u(U_vec::AbstractVector)
            U = @views reshape(U_vec, n_u, H)
            return J(U)
        end
    end

    # Create the sparse Jacobian patterns.
    Jac_pattern_h = Vector{Tuple{Int,Int}}(undef, length(sparsity_Jacobian_rows))
    for i in eachindex(sparsity_Jacobian_rows)
        Jac_pattern_h[i] = (sparsity_Jacobian_rows[i], sparsity_Jacobian_columns[i])
    end

    return PMMH_OCP_Evaluator(K, H, n_u, n_x, n_y,
        n_z, n_h,
        PMMH_samples, V, W, X_t,
        h_dynamics_x!, h_dynamics_y!, h_scenario!, h_u!, h_J_max!, eval_J_u, J_u,
        indices_U, indices_X, indices_Y, index_J_max,
        indices_h_dynamics_x, indices_h_dynamics_y, indices_h_scenario, indices_h_u, indices_h_J_max,
        z_sets, h_bounds,
        Jac_pattern_h,
        Jac_pattern_h_dynamics_x, colors_h_dynamics_x, nzrange_h_dynamics_x,
        Jac_pattern_h_dynamics_y, colors_h_dynamics_y, nzrange_h_dynamics_y,
        Jac_pattern_h_scenario, colors_h_scenario, nzrange_h_scenario,
        Jac_pattern_h_u, colors_h_u, nzrange_h_u,
        Jac_pattern_h_J_max, colors_h_J_max, nzrange_h_J_max)
end

# Wrappers that allow to pack/unpack z by passing the evaluator.
pack_z(U::AbstractMatrix, X::AbstractArray, Y::AbstractArray, e::PMMH_OCP_Evaluator; J_max::Union{Nothing,AbstractFloat}=nothing) = pack_z(U, X, Y, e.indices_U, e.indices_X, e.indices_Y, e.index_J_max, e.K, e.n_z, e.J_u; J_max=J_max)

unpack_z(z::AbstractVector, e::PMMH_OCP_Evaluator) = unpack_z(z, e.indices_U, e.indices_X, e.indices_Y, e.index_J_max, e.K, e.n_u, e.n_x, e.n_y, e.H, e.J_u)

# This function returns the features that are available for the PMMH_OCP_Evaluator; see MathOptInterface documentation.
function MOI.features_available(::PMMH_OCP_Evaluator)
    return [:Grad, :Jac]
end

# The following function is required by MathOptInterface.
# It is called once the evaluator is added to the model.
# This function only checks if the requested features are available.
function MOI.initialize(e::PMMH_OCP_Evaluator, requested::Vector{Symbol})
    available = Set(MOI.features_available(e))
    bad = setdiff(requested, available)

    if !isempty(bad)
        throw(MOI.UnsupportedFeature(
            "Requested feature(s) $(collect(bad)) not supported." * "Available: $(collect(avail))"))
    end
    return nothing
end

# The following function evaluates the objective function.
function MOI.eval_objective(e::PMMH_OCP_Evaluator, z::AbstractVector)
    if e.J_u
        # J(U) is used as the objective function.
        U_vec = view_U_vec(z, e.indices_U)
        return e.eval_J_u(U_vec)
    else
        # Epigraph notation is used.
        return z[first(e.index_J_max)]
    end
end

# The following function evaluates the gradient of the objective function.
function MOI.eval_objective_gradient(e::PMMH_OCP_Evaluator, grad::AbstractVector, z::AbstractVector)
    fill!(grad, 0.0)
    if e.J_u
        # Gradient of J(U) with respect to U.
        U_vec = view_U_vec(z, e.indices_U)
        g_nonzero = @views grad[e.indices_U]
        ForwardDiff.gradient!(g_nonzero, e.J_U_vec, U_vec)
    else
        # Derivative of the epigraph variable J_max.
        grad[first(e.index_J_max)] = 1.0
    end
end

# Evaluate the constraint vector h(z).
function MOI.eval_constraint(e::PMMH_OCP_Evaluator, h::AbstractVector, z::AbstractVector)
    Threads.@threads for k in 1:e.K
        # Get local variables
        if !e.J_u
            U, X_k, Y_k, J_max = unpack_z_k(z, k, e.indices_U, e.indices_X, e.indices_Y, e.index_J_max, e.n_u, e.n_x, e.n_y, e.H, e.J_u)
        else
            U, X_k, Y_k = unpack_z_k(z, k, e.indices_U, e.indices_X, e.indices_Y, e.index_J_max, e.n_u, e.n_x, e.n_y, e.H, e.J_u)
        end

        # Local slices of the constraint vector h
        h_dyn_x = @views h[e.indices_h_dynamics_x[k]]
        h_dyn_y = @views h[e.indices_h_dynamics_y[k]]
        h_scn = @views h[e.indices_h_scenario[k]]

        V_k = @views e.V[:, :, k]
        W_k = @views e.W[:, :, k]

        # Evaluate the dynamic constraints for the states and outputs.
        e.h_dynamics_x!(h_dyn_x, U, X_k, e.PMMH_samples[k].theta, V_k)
        e.h_dynamics_y!(h_dyn_y, U, X_k, Y_k, e.PMMH_samples[k].theta, W_k)

        # Evaluate the scenario constraints.
        e.h_scenario!(h_scn, U, X_k, Y_k)

        # Epigraph rows of constraint vector.
        if !e.J_u
            h_J_max = @views h[e.indices_h_J_max[k]]
            e.h_J_max!(h_J_max, U, X_k, Y_k, J_max)
        end
    end

    # Evaluate the input constraints h_u(U)
    U_vec = view_U_vec(z, e.indices_U)
    h_u = @views h[e.indices_h_u]
    e.h_u!(h_u, U_vec)
    return h
end

# Return the sparsity pattern of the Jacobian of the constraints.
function MOI.jacobian_structure(e::PMMH_OCP_Evaluator)
    return e.Jac_pattern_h
end

# Evaluate the Jacobian of the constraints.
function MOI.eval_constraint_jacobian(e::PMMH_OCP_Evaluator, constraint_Jacobian_values::AbstractVector, z::AbstractVector)

    Threads.@threads for k in 1:e.K
        # Local decision vector
        z_scenario = view_z_scenario(z, k, e.indices_U, e.indices_X, e.indices_Y, e.index_J_max, e.J_u)

        V_k = @views e.V[:, :, k]
        W_k = @views e.W[:, :, k]

        theta_k = e.PMMH_samples[k].theta

        # Initialize.
        Jacobian_h_dynamics_x = Float64.(e.Jac_pattern_h_dynamics_x)
        Jacobian_h_dynamics_y = Float64.(e.Jac_pattern_h_dynamics_y)

        # Compute Jacobian of the dynamic constraints for one scenario.
        SparseDiffTools.forwarddiff_color_jacobian!(Jacobian_h_dynamics_x, (h_dyn_x, z) -> e.h_dynamics_x!(h_dyn_x, z, theta_k, V_k), z_scenario, colorvec=e.colors_h_dynamics_x, sparsity=e.Jac_pattern_h_dynamics_x)
        SparseDiffTools.forwarddiff_color_jacobian!(Jacobian_h_dynamics_y, (h_dyn_y, z) -> e.h_dynamics_y!(h_dyn_y, z, theta_k, W_k), z_scenario, colorvec=e.colors_h_dynamics_y, sparsity=e.Jac_pattern_h_dynamics_y)

        # Fill the values to the vector containing the non-zero entries of the constraint Jacobian.
        constraint_Jacobian_values[e.nzrange_h_dynamics_x[k]] .= Jacobian_h_dynamics_x.nzval
        constraint_Jacobian_values[e.nzrange_h_dynamics_y[k]] .= Jacobian_h_dynamics_y.nzval

        # Compute Jacobian of the scenario constraints for one scenario and fill the values to the vector containing the non-zero entries of the constraint Jacobian.
        Jacobian_h_scenario = Float64.(e.Jac_pattern_h_scenario)
        SparseDiffTools.forwarddiff_color_jacobian!(Jacobian_h_scenario, e.h_scenario!, z_scenario, colorvec=e.colors_h_scenario, sparsity=e.Jac_pattern_h_scenario)
        constraint_Jacobian_values[e.nzrange_h_scenario[k]] .= Jacobian_h_scenario.nzval

        # Compute the Jacobian of the epigraph constraint J^[k] - J_max <= 0 (if used) and fill the values to the vector containing the non-zero entries of the constraint Jacobian.
        if !e.J_u
            Jacobian_h_J_max = Float64.(e.Jac_pattern_h_J_max)
            SparseDiffTools.forwarddiff_color_jacobian!(Jacobian_h_J_max, e.h_J_max!, z_scenario, colorvec=e.colors_h_J_max, sparsity=e.Jac_pattern_h_J_max)
            constraint_Jacobian_values[e.nzrange_h_J_max[k]] .= Jacobian_h_J_max.nzval
        end
    end

    # Evaluate the Jacobian of the input constraints h_u(U) and fill the values to the vector containing the non-zero entries of the constraint Jacobian.
    U_vec = view_U_vec(z, e.indices_U)
    Jacobian_h_u = Float64.(e.Jac_pattern_h_u)
    SparseDiffTools.forwarddiff_color_jacobian!(Jacobian_h_u, e.h_u!, U_vec, colorvec=e.colors_h_u, sparsity=e.Jac_pattern_h_u)
    constraint_Jacobian_values[e.nzrange_h_u] .= Jacobian_h_u.nzval

    return constraint_Jacobian_values
end