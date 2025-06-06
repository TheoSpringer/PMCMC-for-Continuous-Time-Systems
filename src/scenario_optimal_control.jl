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

# This function returns the index ranges for the control inputs u_{1:H}, the states x_{1:H}^{[k]}, and the outputs y_{1:H}^{[k]} of scenarios k = 1, ..., K inside the flat decision vector z.
function get_z_rows(n_u, n_x, n_y, H, K)
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
    return rows_U, rows_X, rows_Y
end

# This function returns the index ranges for dynamic constraints, the scenario constraints h_scenario, and the input constraints h_u inside the flat constraint vector h(z).
function get_h_rows(n_x, n_y, H, K, n_h_scenario, n_h_u)
    rows_dynamic_constraints = Vector{UnitRange{Int}}(undef, K)
    first_dynamic_constraint = 1
    for k in 1:K
        rows_dynamic_constraints[k] = first_dynamic_constraint+(k-1)*(n_x+n_y)*H+1:first_dynamic_constraint+k*(n_x+n_y)*H-1
    end
    first_h_scenario = first_dynamic_constraint + K * (n_x + n_y) * H
    for k in 1:K
        rows_h_scenario[k] = first_h_scenario+(k-1)*n_h_scenario:first_h_scenario+k*n_h_scenario-1
    end
    first_h_u = first_h_scenario + K * n_h_scenario
    rows_h_u = first_h_u:first_h_u+n_h_u-1
    return rows_dynamic_constraints, rows_h_scenario, rows_h_u
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

    # Index ranges for the constraints
    rows_dynamic_constraints::Vector{UnitRange{Int}}
    rows_h_scenario::Vector{UnitRange{Int}}
    rows_h_u::UnitRange{Int}

    # Global sparsity pattern
    sparsity_Jacobian_rows::Vector{Int32}
    sparsity_Jacobian_columns::Vector{Int32}

    # Local sparsity pattern for the dynamics of one scenario


    # Local sparsity pattern for one scenario
    J_pattern_scenario::SparseMatrixCSC{Bool,Int}

    # Sparsity pattern for h_u
    J_pattern_u::SparseMatrixCSC{Bool,Int}
end

# Constructor for PMMH_OCP_Evaluator. The sparsity pattern is built once for a single scenario and then replicated for all K scenarios.
function PMMH_OCP_Evaluator(PMMH_samples, V, W, f_theta, g_theta, J, J_u, h_scenario, h_u, n_u, n_x, n_y, H)
    # Get number of scenarios and the rows in the flat decision vector z.
    K = length(PMMH_samples)
    rows_U, rows_X, rows_Y = get_z_rows(n_u, n_x, n_y, H, K)

    # Get size of scenario constraints and determine the rows in the flat constraint vector h(z).
    n_h_scenario = length(h_scenario(zeros(n_u, H), zeros(n_x, H), zeros(n_y, H)))
    n_h_u = length(h_u(zeros(n_u, H)))
    rows_dynamic_constraints, rows_h_scenario, rows_h_u = get_h_rows(n_x, n_y, H, K, n_h_scenario, n_h_u)

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
        h .= h_scenario(U, X_k, Y_k)
    end

    # Evaluate the sparsity pattern of the Jacobian of h_scenario for a single scenario.
    J_pattern_scenario = Symbolics.jacobian_sparsity(h_scenario_vec, h_scenario_local, z_local)

    # Find the non-zero entries in the sparsity pattern of a single scenario constraint.
    rows_scn, cols_scn, _ = findnz(J_pattern_scenario)

    # Replicate the sparsity pattern over K scenarios.
    sparsity_Jacobian_rows = Int32[] # global row indices of non-zero entries
    sparsity_Jacobian_columns = Int32[] # global column indices of non-zero entries

    # Loop over the scenarios and shift the local sparsity pattern.
    for k in 1:K
        row_offset = first(rows_h_scenario[k]) # first row of scenario k in the global constraint vector
        column_offset_X = first(rows_X[k]) - 1 # first column of the state of scenario k x_{1:H}^{[k]} in the global variable vector
        column_offset_Y = first(rows_Y[k]) - 1 # first column of the output of scenario k y_{1:H}^{[k]} in the global variable vector
        for (r, c) in zip(rows_scn, cols_scn)
            if c < n_u * H
                # Entry belongs to the input block.
                global_column = rows_U[c]
            elseif c < n_u * H + n_x * H
                # Entry belongs to the state block.
                local_rows_X = c - n_u * H
                global_column = column_offset_X + local_rows_X

            else
                # Entry belongs to the output block.
                local_rows_Y = c - n_u * H - n_x * H
                global_column = column_offset_Y + local_rows_Y
            end

            # Add the global row and column indices to the lists.
            push!(sparsity_Jacobian_rows, row_offset + r)
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
        h .= h_u(u)
    end

    # Evaluate the sparsity pattern of h_u.
    J_pattern_u = Symbolics.jacobian_sparsity(h_u_vec, h_u_local, u_local)

    # Find the non-zero entries in the sparsity pattern of h_u.
    rows_u, cols_u, _ = findnz(J_pattern_u)

    row_offset = first(rows_h_u)
    for (r, c) in zip(rows_u, cols_u)
        push!(sparsity_Jacobian_rows, row_offset + r)
        push!(sparsity_Jacobian_columns, rows_U[c])
    end

    sparsity_Jacobian_rows = MOI.RawIndexType.(sparsity_Jacobian_rows)
    sparsity_Jacobian_columns = MOI.RawIndexType.(sparsity_Jacobian_columns)

    # Update this part
    n_h_tot = K * n_h_scenario + n_h_u

    return PMMH_OCP_Evaluator(K, H, n_u, n_x, n_y, n_h_scenario, n_h_u, n_h_tot,
        PMMH_samples, V, W, f_theta, g_theta, J, J_u, h_scenario, h_u,
        rows_U, rows_X, rows_Y, sparsity_Jacobian_rows, sparsity_Jacobian_columns, J_pattern_scenario, J_pattern_u)
end

# ─────────────────────────────────────────────────────────────────────────────
#  MOI interface                                                               
# ─────────────────────────────────────────────────────────────────────────────
MOI.initialize(::PMMH_OCP_Evaluator, _) = nothing

function MOI.eval_objective(e::PMMH_OCP_Evaluator, z::Vector{Float64})
    U = @views reshape(z[e.rows_U], e.n_u, e.H)
    if e.J_u
        return e.J(U)
    else
        tot = zero(Float64)
        for k in 1:e.K
            _, Xk, Yk = extract_views(e.rows_U, e.rows_X[k], e.rows_Y[k], z, e.n_u, e.n_x, e.n_y, e.H)
            tot += e.J(U, Xk, Yk)
        end
        return tot / e.K
    end
end

MOI.eval_gradient(e::PMMH_OCP_Evaluator, g::Vector{Float64}, z::Vector{Float64}) =
    g[:] = ForwardDiff.gradient(v -> MOI.eval_objective(e, v), z)

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
            hk_fun, vars_k, e.J_pattern_scenario)
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