# This file contains the constructor of a custom MathOptInterface evaluator for a scenario-based optimal control problem.
# By exploiting the specific structure of the problem and the sparsity pattern, the evaluator can efficiently compute the objective, the objective gradient, the constraints and the constraint Jacobian.
# The evaluator utilizes thread parallelism to speed up the evaluation of the constraints and their Jacobian.
# To profit from the implementation ensure that multiple threads are available by setting the JULIA_NUM_THREADS environment variable.

# Evaluator struct.
struct PMCMC_OCP_Evaluator <: MOI.AbstractNLPEvaluator
    # Dimensions
    K::Int
    H::Int
    n_u::Int
    n_x::Int
    n_y::Int
    n_z::Int # total number of decision variables
    n_h::Int # total number of constraints

    # Data
    PMCMC_samples::Vector{PMCMC_sample}
    V::Array{Float64,3}
    W::Array{Float64,3}
    X_t::Array{Float64,2} # initial states for the scenarios

    # Functions
    h_dynamics_x!::Function # dynamic constraints for the states
    h_dynamics_y!::Function # dynamic constraints for the outputs
    h_scenario!::Function # scenario constraints
    h_u!::Function # input constraints
    h_J_max!::Function # epigraph constraints (optional)
    eval_J_u::Function # evaluates J(u)
    J_u::Bool

    # Index ranges for the flat decision vector z
    indices_U::UnitRange{Int}
    indices_X::Vector{UnitRange{Int}}
    indices_Y::Vector{UnitRange{Int}}
    index_J_max::UnitRange{Int}

    # Index ranges for the constraint vector h(z)
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

    # Ranges of the constraints in the vector containing the non-zero entries of the global Jacobian.
    nzrange_h_dynamics_x::Vector{UnitRange{Int}}
    nzrange_h_dynamics_y::Vector{UnitRange{Int}}
    nzrange_h_scenario::Vector{UnitRange{Int}}
    nzrange_h_u::UnitRange{Int}
    nzrange_h_J_max::Vector{UnitRange{Int}}

    # Cache for the automatic differentiation
    thread_cache::Vector{ThreadCache}
    cache_h_u::ConstraintCache
end

# Constructor for PMCMC_OCP_Evaluator. The sparsity pattern is built once for a single scenario and then replicated for all K scenarios.
function PMCMC_OCP_Evaluator(PMCMC_samples::Vector{PMCMC_sample}, V::Array{Float64}, W::Array{Float64}, X_t::Array{Float64}, f_theta::Function, g_theta::Function, J::Function, J_u::Bool, h_scenario::Function, h_u::Function, n_u::Int, n_x::Int, n_y::Int, H::Int)
    # Check if multithreading is enabled.
    n_threads = Threads.nthreads()

    @info "Evaluator running with $n_threads Julia thread$(n_threads == 1 ? "" : "s")."

    if n_threads == 1
        @warn "Multithreading is disabled (JULIA_NUM_THREADS = 1).\n" *
              "Jacobian computations and other parallel loops will run serially. " *
              "Enable multithreading for better performance."
    end

    # Get the number of decision variables.
    K = length(PMCMC_samples)

    if !J_u
        n_z = n_u * H + K * n_x * H + K * n_y * H + 1
        n_z_scenario = n_u * H + n_x * H + n_y * H + 1
    else
        n_z = n_u * H + K * n_x * H + K * n_y * H
        n_z_scenario = n_u * H + n_x * H + n_y * H
    end

    # Get the index ranges in the flat decision vector z corresponding to the inputs, states, outputs, and (optionally) J_max.
    indices_U, indices_X, indices_Y, index_J_max = z_indices(n_u, n_x, n_y, H, K, J_u)

    # Get bounds for the flat decision vector z.
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

    # Get the length of the flat constraint vector h(z).
    h_scenario_test = h_scenario(zeros(n_u, H), zeros(n_x, H), zeros(n_y, H))
    if isa(h_scenario_test, AbstractArray)
        n_h_scenario = length(vec(h_scenario_test))
    else
        @error "h_scenario must return an array and not a scalar value."
    end

    h_u_test = h_u(zeros(n_u, H))
    if isa(h_u_test, AbstractArray)
        n_h_u = length(vec(h_u_test))
    else
        @error "h_u must return an array and not a scalar value."
    end

    if !J_u
        n_h = K * n_x * (H - 1) + K * n_y * H + K * n_h_scenario + n_h_u + K
    else
        n_h = K * n_x * (H - 1) + K * n_y * H + K * n_h_scenario + n_h_u
    end

    # Get the index ranges in the flat constraint vector h(z) corresponding to the dynamic constraints, scenario constraints h_scenario, input constraints h_u, and the J - J_max (epigraph) constraints.
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

        # Epigraph (inequality) constraints.
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

    # Build the helper functions that evaluate the constraints.
    h_dynamics_x!, h_dynamics_y!, h_scenario!, h_u!, h_J_max! = build_helpers(f_theta, g_theta, h_scenario, h_u, J, J_u, n_u, n_x, n_y, H)

    # Initialize sparsity pattern for the constraint Jacobian.
    sparsity_Jacobian_rows = Int32[] # global row indices of non-zero entries
    sparsity_Jacobian_columns = Int32[] # global column indices of non-zero entries

    # Compute the sparsity pattern for the Jacobian of the dynamics constraints.
    # The following vectors determine in which region the sparsity pattern is evaluated.
    # Without input dependent branches (e.g., min, max, if) the expression tree is fixed and the sparsity pattern does not depend on the actual values of these vectors.
    z_scenario = zeros(n_z_scenario)
    h_dynamics_x_loc = zeros(n_x * (H - 1))
    h_dynamics_y_loc = zeros(n_y * H)

    # Compute the sparsity pattern of the Jacobian of the dynamics constraints for a single scenario, replicate it for all scenarios, and build the per-thread caches for the automatic differentiation.
    h_dynamics_x_example! = (h_dyn_x, z) -> h_dynamics_x!(h_dyn_x, z, PMCMC_samples[1].theta, V[:, :, 1])
    h_dynamics_y_example! = (h_dyn_y, z) -> h_dynamics_y!(h_dyn_y, z, PMCMC_samples[1].theta, W[:, :, 1])
    nzrange_h_dynamics_x, cache_h_dynamics_x = setup_sparse_Jacobian_cache!(sparsity_Jacobian_rows, sparsity_Jacobian_columns, h_dynamics_x_example!, h_dynamics_x_loc, z_scenario, indices_h_dynamics_x, indices_U, indices_X, indices_Y, index_J_max, n_u, n_x, n_y, H, K)
    nzrange_h_dynamics_y, cache_h_dynamics_y = setup_sparse_Jacobian_cache!(sparsity_Jacobian_rows, sparsity_Jacobian_columns, h_dynamics_y_example!, h_dynamics_y_loc, z_scenario, indices_h_dynamics_x, indices_U, indices_X, indices_Y, index_J_max, n_u, n_x, n_y, H, K)

    # Compute the sparsity pattern for the Jacobian of h_scenario.
    # The following vectors determine in which region the sparsity pattern is evaluated.
    # Without input dependent branches (e.g., min, max, if) the expression tree is fixed and the sparsity pattern does not depend on the actual values of these vectors.
    h_scenario_loc = zeros(n_h_scenario)

    # Compute the sparsity pattern of the Jacobian of the scenario constraints for a single scenario, replicate it for all scenarios, and build the per-thread caches for the automatic differentiation.
    nzrange_h_scenario, cache_h_scenario = setup_sparse_Jacobian_cache!(sparsity_Jacobian_rows, sparsity_Jacobian_columns, h_scenario!, h_scenario_loc, z_scenario, indices_h_scenario, indices_U, indices_X, indices_Y, index_J_max, n_u, n_x, n_y, H, K)

    # Determine the sparsity pattern for the Jacobian of h_u.
    # The following vectors determine in which region the sparsity pattern is evaluated.
    # Without input dependent branches (e.g., min, max, if) the expression tree is fixed and the sparsity pattern does not depend on the actual values of these vectors.
    u_loc = zeros(n_u * H)
    h_u_loc = zeros(n_h_u)

    # Evaluate the sparsity pattern of h_u.
    Jac_pattern_h_u, colors_h_u, rows_h_u_local, columns_h_u_local = compute_jacobian_sparsity(h_u!, u_loc, h_u_loc)

    # Add the non-zero entries of the sparsity pattern of h_u to the global sparsity pattern.
    row_offset = first(indices_h_u) - 1
    start = length(sparsity_Jacobian_rows) + 1
    for (r, c) in zip(rows_h_u_local, columns_h_u_local)
        push!(sparsity_Jacobian_rows, row_offset + r)
        push!(sparsity_Jacobian_columns, indices_U[c])
    end
    stop = length(sparsity_Jacobian_rows)
    nzrange_h_u = start:stop

    # Build a cache for the automatic differentiation of h_u.
    cache_h_u = ConstraintCache(SparseDiffTools.ForwardColorJacCache(h_u!, u_loc, nothing; dx=h_u_loc, colorvec=colors_h_u, sparsity=Jac_pattern_h_u), Float64.(Jac_pattern_h_u))

    if !J_u
        # Evaluate the sparsity pattern of the Jacobian of the epigraph constraints.
        # The following vectors determine in which region the sparsity pattern is evaluated.
        h_J_max_loc = zeros(1)

        # Compute the sparsity pattern of the Jacobian of the scenario constraints for a single scenario, replicate it for all scenarios, and build the per-thread caches for the automatic differentiation.
        nzrange_h_J_max, cache_h_J_max = setup_sparse_Jacobian_cache!(sparsity_Jacobian_rows, sparsity_Jacobian_columns, h_J_max!, h_J_max_loc, z_scenario, indices_h_scenario, indices_U, indices_X, indices_Y, index_J_max, n_u, n_x, n_y, H, K)

        eval_J_u = (U_vec) -> nothing
    else
        nzrange_h_J_max = 1:0
        cache_h_J_max = nothing

        # To compute the gradient of J(U), we need to define a vectorized version of J(U).
        function eval_J_u(U_vec::AbstractVector)
            U = @views reshape(U_vec, n_u, H)
            return J(U)
        end

        #=
        # Determine the sparsity pattern of the Jacobian of J(U).
        u = zeros(n_u * H)
        h_J_u_loc = zeros(1)

        Jac_pattern_J_u, colors_J_u, _, _ = compute_jacobian_sparsity(eval_J_u, u, h_J_u_loc)

        # Create a cache for the automatic differentiation of J(U).
        cache_J_u = ConstraintCache(SparseDiffTools.ForwardColorJacCache(eval_J_u, u, nothing; dx=h_J_u_loc, colorvec=colors_J_u, sparsity=Jac_pattern_J_u), Float64.(Jac_pattern_J_u))
        =#
    end

    # Create the cache for the automatic differentiation of the constraints for each thread.
    thread_cache = Vector{ThreadCache}(undef, n_threads)
    for i in 1:n_threads
        thread_cache[i] = ThreadCache(cache_h_dynamics_x, cache_h_dynamics_y, cache_h_scenario, cache_h_J_max)
    end

    # Quick check to ensure the sparsity patterns are valid.
    @assert maximum(sparsity_Jacobian_rows) <= n_h
    @assert maximum(sparsity_Jacobian_columns) <= n_z

    # Create the sparse Jacobian patterns.
    Jac_pattern_h = Vector{Tuple{Int,Int}}(undef, length(sparsity_Jacobian_rows))
    for i in eachindex(sparsity_Jacobian_rows)
        Jac_pattern_h[i] = (sparsity_Jacobian_rows[i], sparsity_Jacobian_columns[i])
    end

    return PMCMC_OCP_Evaluator(K, H, n_u, n_x, n_y,
        n_z, n_h,
        PMCMC_samples, V, W, X_t,
        h_dynamics_x!, h_dynamics_y!, h_scenario!, h_u!, h_J_max!, eval_J_u, J_u,
        indices_U, indices_X, indices_Y, index_J_max,
        indices_h_dynamics_x, indices_h_dynamics_y, indices_h_scenario, indices_h_u, indices_h_J_max,
        z_sets, h_bounds,
        Jac_pattern_h,
        nzrange_h_dynamics_x, nzrange_h_dynamics_y, nzrange_h_scenario, nzrange_h_u, nzrange_h_J_max,
        thread_cache, cache_h_u)
end