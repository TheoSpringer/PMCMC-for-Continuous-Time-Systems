using LinearAlgebra
using Sybmbolics
using SparseArrays
using ForwardDiff
using SparseDiffTools
using Threads
using JuMP
using Ipopt
using MathOptInterface
const MOI = MathOptInterface

# This function returns the index ranges for the control inputs u_{1:H}, the states x_{1:H}^{[k]} of scenarios 1:K, the outputs y_{1:H}^{[k]} of scenarios 1:K, and the maximum cost J_max inside the flat decision vector z.
function get_z_rows(n_u, n_x, n_y, H, K, J_u)
    rows_U = 1:(n_u*H)
    rows_X = Vector{UnitRange{Int}}(undef, K)
    rows_Y = Vector{UnitRange{Int}}(undef, K)
    first_X = last(rows_U) + 1
    for k in 1:K
        rows_X[k] = first_X+(k-1)*n_x*H:first_X+k*n_x*H-1
    end
    first_Y = first_X + K * n_x * H
    for k in 1:K
        rows_Y[k] = first_Y+(k-1)*n_y*H:first_Y+k*n_y*H-1
    end
    if J_u
        # Pass empty range if J_u is true
        row_J_max = 1:0
    else
        row_J_max = first_Y+K*n_y*H:first_Y+K*n_y*H
    end
    return rows_U, rows_X, rows_Y, row_J_max
end

# This function returns the index ranges for dynamic constraints, the scenario constraints h_scenario, and the input constraints h_u inside the flat constraint vector h(z).
function get_h_rows(n_x, n_y, H, K, n_h_scenario, n_h_u, J_u)
    rows_dynamic_constraints_x = Vector{UnitRange{Int}}(undef, K)
    first_dynamic_constraint_x = 1
    for k in 1:K
        rows_dynamic_constraints_x[k] = first_dynamic_constraint_x+(k-1)*n_x*(H-1)+1:first_dynamic_constraint_x+k*n_x*(H-1)-1
    end
    rows_dynamic_constraints_y = Vector{UnitRange{Int}}(undef, K)
    first_dynamic_constraint_y = first_dynamic_constraint_x + k * n_x * (H - 1)
    for k in 1:K
        rows_dynamic_constraints_y[k] = first_dynamic_constraint_y+(k-1)*n_y*H+1:first_dynamic_constraint_y+k*n_y*H-1
    end
    first_h_scenario = first_dynamic_constraint_y + K * n_y * H
    for k in 1:K
        rows_h_scenario[k] = first_h_scenario+(k-1)*n_h_scenario:first_h_scenario+k*n_h_scenario-1
    end
    first_h_u = first_h_scenario + K * n_h_scenario
    rows_h_u = first_h_u:first_h_u+n_h_u-1
    if J_u
        # Pass empty range if J_u is true
        rows_J_max_constraints = 1:0
    else
        rows_J_max_constraints = first_h_u+n_h_u:first_h_u+n_h_u+K-1
    end
    return rows_dynamic_constraints_x, rows_dynamic_constraints_y, rows_h_scenario, rows_h_u, rows_J_max_constraints
end

# Zero‑copy extract views.
@inline function extract_views(rows_U, rows_Xk, rows_Yk, z, n_u, n_x, n_y, H)
    @views U = reshape(z[rows_U], n_u, H)
    @views Xk = reshape(z[rows_Xk], n_x, H)
    @views Yk = reshape(z[rows_Yk], n_y, H)
    return U, Xk, Yk
end

# Evaluator struct.
struct PMMH_OCP_Evaluator <: MOI.AbstractNLPEvaluator
    # Dimensions
    K::Int
    H::Int
    n_u::Int
    n_x::Int
    n_y::Int

    # Data
    PMMH_samples::Vector{PMMH_sample}
    V::Array{Float64,3}
    W::Array{Float64,3}

    # Functions
    f_theta::Function
    g_theta::Function
    J::Function
    J_u::Bool
    h_scenario::Function
    h_u::Function

    # Index ranges 
    rows_U::UnitRange{Int}
    rows_X::Vector{UnitRange{Int}}
    rows_Y::Vector{UnitRange{Int}}
    row_J_max::UnitRange{Int}

    # Index ranges for the constraints
    rows_dynamic_constraints_x::Vector{UnitRange{Int}}
    rows_dynamic_constraints_y::Vector{UnitRange{Int}}
    rows_h_scenario::Vector{UnitRange{Int}}
    rows_h_u::UnitRange{Int}
    rows_J_max_constraints::UnitRange{Int}

    # Global sparsity pattern
    sparsity_Jacobian_rows::Vector{Int32}
    sparsity_Jacobian_columns::Vector{Int32}

    # Local sparsity pattern for the dynamics constraints of one scenario
    Jac_pattern_dynamic_constraints_x::SparseMatrixCSC{Bool,Int}
    Jac_pattern_dynamic_constraints_y::SparseMatrixCSC{Bool,Int}

    # Local sparsity pattern for the scenario constraints h_scenario of one scenario
    Jac_pattern_h_scenario::SparseMatrixCSC{Bool,Int}

    # Sparsity pattern for h_u
    Jac_pattern_h_u::SparseMatrixCSC{Bool,Int}

    # Local sparsity pattern for the J - J_max constraint
    Jac_pattern_J_max::SparseMatrixCSC{Bool,Int}
end

# Constructor for PMMH_OCP_Evaluator. The sparsity pattern is built once for a single scenario and then replicated for all K scenarios.
function PMMH_OCP_Evaluator(PMMH_samples, V, W, f_theta, g_theta, J, J_u, h_scenario, h_u, n_u, n_x, n_y, H)
    # Get number of scenarios and the rows in the flat decision vector z.
    K = length(PMMH_samples)
    rows_U, rows_X, rows_Y, row_J_max = get_z_rows(n_u, n_x, n_y, H, K)

    # Get size of scenario constraints and determine the rows in the flat constraint vector h(z).
    n_h_scenario = length(h_scenario(zeros(n_u, H), zeros(n_x, H), zeros(n_y, H))[:])
    n_h_u = length(h_u(zeros(n_u, H))[:])
    rows_dynamic_constraints_x, rows_dynamic_constraints_y, rows_h_scenario, rows_h_u, rows_J_max_constraints = get_h_rows(n_x, n_y, H, K, n_h_scenario, n_h_u)

    # Determine the sparsity pattern for the Jacobian of the dynamics constraints.
    # The following vectors determine in which region the sparsity pattern is evaluated.
    # Without input dependent branches (e.g., min, max, if) the expression tree is fixed and the sparsity pattern does not depend on the actual values of these vectors.
    z_local = zeros(n_u * H + n_x * H + n_y * H)
    h_dynamics_x_local = zeros(n_x * (H - 1))
    h_dynamics_y_local = zeros(n_y * H)

    # Helper function that evaluates the dynamics constraints over the whole horizon H for one scenario with a vector valued input.
    function f_theta_constraint_vec!(F, z)
        U = reshape(@view z[1:n_u*H], n_u, H)
        X_k = reshape(@view z[n_u*H+1:n_u*H+n_x*H], n_x, H)
        for t in 1:H-1
            # The process noise is not included since it does not affect the sparsity pattern.
            F[(t-1)*n_x+1:t*n_x-1] = f_theta(U[:, t], X_k[:, t]) - X_k[:, t+1]
        end
    end

    # Helper function that evaluates the measurement constraints over the whole horizon H for one scenario with a vector valued input.
    function g_theta_constraint_vec!(G, z)
        U = reshape(@view z[1:n_u*H], n_u, H)
        X_k = reshape(@view z[n_u*H+1:n_u*H+n_x*H], n_x, H)
        Y_k = reshape(@view z[n_u*H+n_x*H+1:end], n_y, H)
        for t in 1:H
            # The measurement noise is not included since it does not affect the sparsity pattern.
            G[(t-1)*n_y+1:t*n_y] = g_theta(U[:, t], X_k[:, t]) - Y_k[:, t]
        end
    end

    # Evaluate the sparsity pattern of the Jacobian of F_theta and G_theta for a single scenario.
    Jac_pattern_dynamic_constraints_x = Symbolics.jacobian_sparsity(f_theta_constraint_vec!, h_dynamics_x_local, z_local)
    Jac_pattern_dynamic_constraints_y = Symbolics.jacobian_sparsity(g_theta_constraint_vec!, h_dynamics_y_local, z_local)

    # Find the non-zero entries in the sparsity pattern of the dynamics constraints.
    rows_x, cols_x, _ = findnz(Jac_pattern_dynamic_constraints_x)
    rows_y, cols_y, _ = findnz(Jac_pattern_dynamic_constraints_y)

    # Replicate the sparsity pattern over K scenarios.
    sparsity_Jacobian_rows = Int32[] # global row indices of non-zero entries
    sparsity_Jacobian_columns = Int32[] # global column indices of non-zero entries

    # Loop over the scenarios and shift the local sparsity pattern.
    for k in 1:K
        row_offset_x = first(rows_dynamic_constraints_x[k]) # first row of the dynamic constraints of scenario k in the global constraint vector
        row_offset_y = first(rows_dynamic_constraints_y[k]) # first row of the measurement constraints of scenario k in the global constraint vector
        column_offset_X = first(rows_X[k]) - 1 # first column of the state x_{1:H}^{[k]} of scenario k in the global variable vector
        column_offset_Y = first(rows_Y[k]) - 1 # first column of the output y_{1:H}^{[k]} of scenario k in the global variable vector
        for (r, c) in zip(rows_x, cols_x)
            global_row = row_offset_x + r

            if c <= n_u * H
                # Entry belongs to the input block.
                global_column = rows_U[c]
            elseif c <= n_u * H + n_x * H
                # Entry belongs to the state block.
                local_rows_X = c - n_u * H
                global_column = column_offset_X + local_rows_X

            else
                # Entry belongs to the output block.
                local_rows_Y = c - n_u * H - n_x * H
                global_column = column_offset_Y + local_rows_Y
            end

            # Add the global row and column indices to the lists.
            push!(sparsity_Jacobian_rows, global_row)
            push!(sparsity_Jacobian_columns, global_column)
        end

        for (r, c) in zip(rows_y, cols_y)
            global_row = row_offset_y + r

            if c <= n_u * H
                # Entry belongs to the input block.
                global_column = rows_U[c]
            elseif c <= n_u * H + n_x * H
                # Entry belongs to the state block.
                local_rows_X = c - n_u * H
                global_column = column_offset_X + local_rows_X

            else
                # Entry belongs to the output block.
                local_rows_Y = c - n_u * H - n_x * H
                global_column = column_offset_Y + local_rows_Y
            end

            # Add the global row and column indices to the lists.
            push!(sparsity_Jacobian_rows, global_row)
            push!(sparsity_Jacobian_columns, global_column)
        end
    end

    # Determine the sparsity pattern for the Jacobian of h_scenario.
    # The following vectors determine in which region the sparsity pattern is evaluated.
    # Without input dependent branches (e.g., min, max, if) the expression tree is fixed and the sparsity pattern does not depend on the actual values of these vectors.
    z_local = zeros(n_u * H + n_x * H + n_y * H)
    h_scenario_local = zeros(n_h_scenario)

    # Helper function that evaluates constraints for a single scenario with a vector valued input.
    function h_scenario_vec!(h, z)
        U = reshape(@view z[1:n_u*H], n_u, H)
        X_k = reshape(@view z[n_u*H+1:n_u*H+n_x*H], n_x, H)
        Y_k = reshape(@view z[n_u*H+n_x*H+1:end], n_y, H)
        h .= h_scenario(U, X_k, Y_k)[:]
    end

    # Evaluate the sparsity pattern of the Jacobian of h_scenario for a single scenario.
    Jac_pattern_h_scenario = Symbolics.jacobian_sparsity(h_scenario_vec!, h_scenario_local, z_local)

    # Find the non-zero entries in the sparsity pattern of a single scenario constraint.
    rows_scn, cols_scn, _ = findnz(Jac_pattern_h_scenario)

    # Loop over the scenarios and shift the local sparsity pattern.
    for k in 1:K
        row_offset = first(rows_h_scenario[k]) # first row of h_scenario() of scenario k in the global constraint vector
        column_offset_X = first(rows_X[k]) - 1 # first column of the state x_{1:H}^{[k]} of scenario k in the global variable vector
        column_offset_Y = first(rows_Y[k]) - 1 # first column of the output y_{1:H}^{[k]} of scenario k in the global variable vector
        for (r, c) in zip(rows_scn, cols_scn)
            global_row = row_offset + r

            if c <= n_u * H
                # Entry belongs to the input block.
                global_column = rows_U[c]
            elseif c <= n_u * H + n_x * H
                # Entry belongs to the state block.
                local_rows_X = c - n_u * H
                global_column = column_offset_X + local_rows_X

            else
                # Entry belongs to the output block.
                local_rows_Y = c - n_u * H - n_x * H
                global_column = column_offset_Y + local_rows_Y
            end

            # Add the global row and column indices to the lists.
            push!(sparsity_Jacobian_rows, global_row)
            push!(sparsity_Jacobian_columns, global_column)
        end
    end

    # Get size of input constraints.
    n_h_u = length(h_u(zeros(n_u, H)))

    # Determine the sparsity pattern for the Jacobian of h_u.
    # The following vectors determine in which region the sparsity pattern is evaluated.
    # Without input dependent branches (e.g., min, max, if) the expression tree is fixed and the sparsity pattern does not depend on the actual values of these vectors.
    u_local = zeros(n_u * H)
    h_u_local = zeros(n_h_u)

    # Helper function that evaluates h_u with a vector valued input.
    function h_u_vec!(h, u)
        h .= h_u(u)[:]
    end

    # Evaluate the sparsity pattern of h_u.
    Jac_pattern_h_u = Symbolics.jacobian_sparsity(h_u_vec!, h_u_local, u_local)

    # Find the non-zero entries in the sparsity pattern of h_u.
    rows_u, cols_u, _ = findnz(Jac_pattern_h_u)

    row_offset = first(rows_h_u)
    for (r, c) in zip(rows_u, cols_u)
        push!(sparsity_Jacobian_rows, row_offset + r)
        push!(sparsity_Jacobian_columns, rows_U[c])
    end

    if !J_u
        # Evaluate the sparsity pattern of the Jacobian of J^[k] - J_max (constraint utilized to minimize the worst-case cost).
        # The following vectors determine in which region the sparsity pattern is evaluated.
        z_local = zeros(n_u * H + n_x * H + n_y * H)
        J_local = 0.0

        function J_max_constraint_vec!(J_diff, z)
            U = reshape(@view z[1:n_u*H], n_u, H)
            X_k = reshape(@view z[n_u*H+1:n_u*H+n_x*H], n_x, H)
            Y_k = reshape(@view z[n_u*H+n_x*H+1:end-1], n_y, H)
            J_max = z[end]
            J_diff .= J(U, X_k, Y_k) - J_max
        end

        # Evaluate the sparsity pattern of the Jacobian of h_scenario for a single scenario.
        Jac_pattern_J_max = Symbolics.jacobian_sparsity(J_max_constraint_vec!, J_local, z_local)

        # Find the non-zero entries in the sparsity pattern of a single constraint.
        _, cols_J_max, _ = findnz(Jac_pattern_J_max)

        # Loop over the scenarios and shift the local sparsity pattern.
        for k in 1:K
            global_row = rows_J_max_constraints[k] # first row of h_scenario() of scenario k in the global constraint vector
            column_offset_X = first(rows_X[k]) - 1 # first column of the state x_{1:H}^{[k]} of scenario k in the global variable vector
            column_offset_Y = first(rows_Y[k]) - 1 # first column of the output y_{1:H}^{[k]} of scenario k in the global variable vector
            for c in cols_J_max
                if c <= n_u * H
                    # Entry belongs to the input block.
                    global_column = rows_U[c]
                elseif c <= n_u * H + n_x * H
                    # Entry belongs to the state block.
                    local_rows_X = c - n_u * H
                    global_column = column_offset_X + local_rows_X

                elseif c <= n_u * H + n_x * H + n_y * H
                    # Entry belongs to the output block.
                    local_rows_Y = c - n_u * H - n_x * H
                    global_column = column_offset_Y + local_rows_Y
                else
                    # Entry belongs to the J_max constraint.
                    global_column = first(row_J_max)
                end
                # Add the global row and column indices to the lists.
                push!(sparsity_Jacobian_rows, global_row)
                push!(sparsity_Jacobian_columns, global_column)
            end
        end
    end

    sparsity_Jacobian_rows = MOI.RawIndexType.(sparsity_Jacobian_rows)
    sparsity_Jacobian_columns = MOI.RawIndexType.(sparsity_Jacobian_columns)

    return PMMH_OCP_Evaluator(K, H, n_u, n_x, n_y,
        PMMH_samples, V, W, f_theta, g_theta, J, J_u, h_scenario, h_u,
        rows_U, rows_X, rows_Y, row_J_max,
        rows_dynamic_constraints_x, rows_dynamic_constraints_y,
        rows_h_scenario, rows_h_u, rows_J_max_constraints,
        sparsity_Jacobian_rows, sparsity_Jacobian_columns,
        Jac_pattern_dynamic_constraints_x, Jac_pattern_dynamic_constraints_y,
        Jac_pattern_h_scenario, Jac_pattern_h_u, Jac_pattern_J_max)
end

# The following function is required by MathOptInterface.
# It is called once the evaluator is added to the model.
MOI.initialize(::PMMH_OCP_Evaluator, _) = nothing

# The following function evaluates the objective function.
function MOI.eval_objective(e::PMMH_OCP_Evaluator, z::Vector{Float64})
    if e.J_u
        U = @views reshape(z[e.rows_U], e.n_u, e.H)
        return e.J(U)
    else
        return z(e.row_J_max)
    end
end

# The following function evaluates the gradient of the objective function.
function MOI.eval_gradient(e::PMMH_OCP_Evaluator, g::Vector{Float64}, z::Vector{Float64})
    g[:] = zeros(length(z))
    if e.J_u
        # Gradient of J(U) with respect to U
        U = @views reshape(z[e.rows_U], e.n_u, e.H)
        g[rows_U] = ForwardDiff.gradient(U -> e.J(U), U)
    else
        g[e.row_J_max] = 1.0
    end
    return g
end

function MOI.eval_constraint(e::PMMH_OCP_Evaluator, g::Vector{Float64}, z::Vector{Float64})
    pos = 1
    for k in 1:e.K
        U, Xk, Yk = extract_views(e.rows_U, e.rows_X[k], e.rows_Y[k], z, e.n_u, e.n_x, e.n_y, e.H)
        hk = e.h_scenario(U, Xk, Yk)
        @inbounds g[pos:pos+e.n_h_scenario-1] = hk
        pos += e.n_h_scenario
    end
    g[pos:end] = e.h_u(@views reshape(z[e.rows_U], e.n_u, e.H))
end

MOI.jacobian_structure(e::PMMH_OCP_Evaluator) = (e.sparsity_row_index, e.sparsity_column_index)

function MOI.eval_jacobian(e::PMMH_OCP_Evaluator, Jval::Vector{Float64}, z::Vector{Float64})
    # Thread parallel: one scenario per thread
    Threads.@threads for k in 1:e.K
        # Local slices
        local_rows = (k - 1) * e.n_h_scenario * length(z) + 1   # not used (faster: write view)
        vars_k = vcat(@views z[e.rows_U], @views z[e.rows_X[k]], @views z[e.rows_Y[k]])
        hk_fun(v) = e.h_scenario(@views reshape(v[1:e.n_u*e.H], e.n_u, e.H),
        @views reshape(v[e.n_u*e.H+1:e.n_u*e.H+e.n_x*e.H], e.n_x, e.H),
        @views reshape(v[e.n_u*e.H+e.n_x*e.H+1:end], e.n_y, e.H))
        local_J = SparseDiffTools.forwarddiff_color_jacobian(
            hk_fun, vars_k, e.Jac_pattern_h_scenario)
        # copy into global Jval
        global_row_offset = (k - 1) * e.n_h_scenario
        for (nz_i, (r, c, v)) in enumerate(zip(local_J.row .- 1, local_J.col .- 1, local_J.nzval))
            global_r = global_row_offset + r
            global_column = c < e.n_u * e.H ? e.rows_U[c+1] - 1 :
                            c < e.n_u * e.H + e.n_x * e.H ? first(e.rows_X[k]) - 1 + c - e.n_u * e.H :
                            first(e.rows_Y[k] - 1 + c - e.n_u * e.H - e.n_x * e.H)
            # linear index into compressed vector (same order as sparsity_row_index,sparsity_column_index)
            lin = findfirst((e.sparsity_row_index .== global_r) .& (e.sparsity_column_index .== global_column))
            Jval[lin] = v
        end
    end
    # rows due to h_u(U) — fill with AD once (small); omitted for brevity
    return
end

# ─────────────────────────────────────────────────────────────────────────────
#  Driver                                                                      
# ─────────────────────────────────────────────────────────────────────────────
function solve_PMMH_OCP_parallel(PMMH_samples::Vector{PMMH_sample}, n_y::Int,
    f_theta::Function, g_theta::Function, sample_v_theta::Function,
    sample_w_theta::Function, H, J::Function, h_scenario::Function, h_u::Function;
    J_u=false, V=nothing, W=nothing, print_progress=true)

    # dimensions
    K = length(PMMH_samples)
    n_u = size(PMMH_samples[1].u_m1, 1)
    n_x = size(PMMH_samples[1].x_m1, 1)

    V = V === nothing ? zeros(n_x, H, K) : V
    W = W === nothing ? zeros(n_y, H, K) : W

    # evaluator
    eval = PMMH_OCP_Evaluator(PMMH_samples, V, W, f_theta, g_theta,
        J, J_u, h_scenario, h_u, n_u, n_x, n_y, H)

    n_var = last(eval.rows_Y[end])
    model = Model(Ipopt.Optimizer)
    backend = backend(model)
    MOI.set(backend, MOI.NLPBlock(),
        MOI.NLPBlockData(eval, true, n_var, eval.m_tot))

    @variable(model, z[1:n_var], start = zeros(n_var))

    # variable bounds example (u in [-5,5])
    for j in eval.rows_U
        set_lower_bound(z[j], -5.0)
        set_upper_bound(z[j], 5.0)
    end

    optimize!(model)
    return termination_status(model)
end