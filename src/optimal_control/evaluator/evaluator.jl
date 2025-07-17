# This file contains the constructor for a custom evaluator for a scenario-based optimal control problem.
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
    h_J_max!::Union{Function,Nothing} # epigraph constraints (optional)
    eval_J_u::Union{Function,Nothing} # evaluates J(u)
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
    indices_h_J_max::Union{Vector{UnitRange{Int}},Nothing}

    # Bounds for the decision vector z
    z_sets::Vector{MOI.AbstractScalarSet}

    # Bounds for the constraint vector h(z)
    h_bounds::Vector{MOI.NLPBoundsPair}

    # Global sparsity pattern of the constraint Jacobian.
    Jacobian_pattern_h::Vector{Tuple{Int,Int}}

    # Global sparsity pattern of the Hessian of the Lagrangian.
    Hessian_pattern_L::Vector{Tuple{Int,Int}}

    # Ranges of the constraints in the vector containing the non-zero entries of the global Jacobian.
    nzrange_h_dynamics_x::Vector{UnitRange{Int}}
    nzrange_h_dynamics_y::Vector{UnitRange{Int}}
    nzrange_h_scenario::Vector{UnitRange{Int}}
    nzrange_h_u::UnitRange{Int}
    nzrange_h_J_max::Union{Vector{UnitRange{Int}},Nothing}

    # Ranges of the Hessian of the local Lagrangian in the vector containing the non-zero entries of the global Hessian.
    nzrange_Hessian_L_dynamics_x::Union{Vector{UnitRange{Int}},Nothing}
    nzrange_Hessian_L_dynamics_y::Union{Vector{UnitRange{Int}},Nothing}
    nzrange_Hessian_L_scenario::Union{Vector{UnitRange{Int}},Nothing}
    nzrange_Hessian_L_u::Union{Vector{UnitRange{Int}},Nothing}
    nzrange_Hessian_L_J_max::Union{Vector{UnitRange{Int}},Nothing}
    nzrange_Hessian_J_u::Union{Vector{UnitRange{Int}},Nothing}

    # Cache for the automatic differentiation
    thread_cache::Vector{ThreadCache}
    cache_Jacobian_h_u::ADCache
    cache_Hessian_L_u::Union{ADCache,Nothing}
    cache_Hessian_J_u::Union{ADCache,Nothing}
end

# Constructor for PMCMC_OCP_Evaluator.
function PMCMC_OCP_Evaluator(PMCMC_samples::Vector{PMCMC_sample}, V::Array{Float64}, W::Array{Float64}, X_t::Array{Float64}, f_theta::Function, g_theta::Function, J::Function, J_u::Bool, h_scenario::Function, h_u::Function, n_u::Int, n_x::Int, n_y::Int, H::Int; build_hessian::Bool=true)
    # Check if multithreading is enabled.
    n_threads = Threads.nthreads()

    @info "Evaluator running with $n_threads Julia thread$(n_threads == 1 ? "" : "s")."

    if n_threads == 1
        @warn "Multithreading is disabled (JULIA_NUM_THREADS = 1).\n" *
              "Jacobian computations and other parallel loops will run serially. " *
              "Enable multithreading for better performance."
    end

    # Get the total number of decision variables.
    K = length(PMCMC_samples)

    if !J_u
        n_z = n_u * H + K * n_x * H + K * n_y * H + 1
        n_z_scenario = n_u * H + n_x * H + n_y * H + 1
    else
        n_z = n_u * H + K * n_x * H + K * n_y * H
        n_z_scenario = n_u * H + n_x * H + n_y * H
    end

    # Get the index ranges in the flat decision vector z corresponding to the inputs U, states X, outputs Y, and (optionally) J_max.
    indices_U, indices_X, indices_Y, index_J_max = z_indices(n_u, n_x, n_y, H, K, J_u)

    # Create bounds for the flat decision vector z.
    z_sets = Vector{MOI.AbstractScalarSet}(undef, n_z)
    for i in indices_U
        z_sets[i] = MOI.Interval(-Inf, Inf)
    end
    for k in 1:K
        for (index_x, i) in enumerate(indices_X[k])
            if i <= n_x
                # Initial state x_1^[k] is fixed.
                z_sets[indices_X[k][index_x]] = MOI.EqualTo(X_t[index_x, k])
            else
                z_sets[i] = MOI.Interval(-Inf, Inf)
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

    # Get the index ranges in the flat constraint vector h(z) corresponding to the dynamic constraints for the states, 
    # the dynamic constraints for the outputs, scenario constraints h_scenario, the input constraints h_u, and (optionally) the epigraph constraints.
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

    # Build the helper functions that evaluate the constraints and the local Lagrangian terms.
    h_dynamics_x!, h_dynamics_y!, h_scenario!, h_u!, h_J_max!, lagrangian_dynamics_x, lagrangian_dynamics_y, lagrangian_scenario, lagrangian_u, lagrangian_J_max = build_helpers(f_theta, g_theta, h_scenario, h_u, J, J_u, n_u, n_x, n_y, H; build_hessian)

    # Initialize sparsity pattern for the constraint Jacobian.
    sparsity_Jacobian_rows = Int32[] # global row indices of non-zero entries
    sparsity_Jacobian_columns = Int32[] # global column indices of non-zero entries

    # Compute the sparsity pattern for the Jacobian of the dynamic constraints.
    # The following vectors determine in which region the sparsity pattern is evaluated.
    # Without input dependent branches (e.g., min, max, if) the expression tree is fixed and the sparsity pattern does not depend on the actual values of these vectors.
    z_scenario_loc = zeros(n_z_scenario)
    h_dynamics_x_loc = zeros(n_x * (H - 1))
    h_dynamics_y_loc = zeros(n_y * H)

    # Compute the sparsity pattern of the Jacobian of the dynamic constraints for a single scenario, replicate it for all scenarios, and build per-thread cache for the automatic differentiation.
    h_dynamics_x_example! = (h_dyn_x, z_scenario) -> h_dynamics_x!(h_dyn_x, z_scenario, PMCMC_samples[1].theta, V[:, :, 1])
    h_dynamics_y_example! = (h_dyn_y, z_scenario) -> h_dynamics_y!(h_dyn_y, z_scenario, PMCMC_samples[1].theta, W[:, :, 1])
    nzrange_h_dynamics_x, cache_h_dynamics_x = setup_sparse_Jacobian_cache!(sparsity_Jacobian_rows, sparsity_Jacobian_columns, h_dynamics_x_example!, h_dynamics_x_loc, z_scenario_loc, indices_h_dynamics_x, indices_U, indices_X, indices_Y, index_J_max, n_u, n_x, n_y, H, K)
    nzrange_h_dynamics_y, cache_h_dynamics_y = setup_sparse_Jacobian_cache!(sparsity_Jacobian_rows, sparsity_Jacobian_columns, h_dynamics_y_example!, h_dynamics_y_loc, z_scenario_loc, indices_h_dynamics_x, indices_U, indices_X, indices_Y, index_J_max, n_u, n_x, n_y, H, K)

    # Compute the sparsity pattern of the Jacobian of the scenario constraints for a single scenario, replicate it for all scenarios, and build per-thread cache for the automatic differentiation.
    h_scenario_loc = zeros(n_h_scenario)
    nzrange_h_scenario, cache_h_scenario = setup_sparse_Jacobian_cache!(sparsity_Jacobian_rows, sparsity_Jacobian_columns, h_scenario!, h_scenario_loc, z_scenario_loc, indices_h_scenario, indices_U, indices_X, indices_Y, index_J_max, n_u, n_x, n_y, H, K)

    # Evaluate the sparsity pattern of h_u.
    u_loc = zeros(n_u * H)
    h_u_loc = zeros(n_h_u)
    Jac_pattern_h_u, colors_h_u, rows_h_u_local, columns_h_u_local = compute_Jacobian_sparsity(h_u!, u_loc, h_u_loc)

    # Add the non-zero entries of the sparsity pattern of h_u to the global sparsity pattern.
    row_offset = first(indices_h_u) - 1
    start = length(sparsity_Jacobian_rows) + 1
    for (r, c) in zip(rows_h_u_local, columns_h_u_local)
        push!(sparsity_Jacobian_rows, row_offset + r)
        push!(sparsity_Jacobian_columns, indices_U[c])
    end
    stop = length(sparsity_Jacobian_rows)
    nzrange_h_u = start:stop

    # Build cache for the automatic differentiation of h_u.
    cache_Jacobian_h_u = ADCache(SparseDiffTools.ForwardColorJacCache(h_u!, u_loc, nothing; dx=h_u_loc, colorvec=colors_h_u, sparsity=Jac_pattern_h_u), Float64.(Jac_pattern_h_u))

    if !J_u
        # Epigraph constraints are used.
        # Compute the sparsity pattern of the Jacobian of the epigraph constraint for a single scenario, replicate it for all scenarios, and build per-thread cache for the automatic differentiation.
        # The epigraph constraint for one scenario is scalar, so its Jacobian is a single row (i.e., the gradient).  With only one row, matrix colouring offers no benefit: we could build the sparsity pattern with
        # `Symbolics.jacobian_sparsity` once and then evaluate the row via `ForwardDiff.gradient!` in the callback.
        # Instead, we deliberately reuse the same Jacobian‑building path we use for the larger constraint blocks.  
        # This keeps the implementation uniform and the callback logic simple, at the cost of a negligible amount of extra work for this one row.
        h_J_max_loc = zeros(1)
        nzrange_h_J_max, cache_h_J_max = setup_sparse_Jacobian_cache!(sparsity_Jacobian_rows, sparsity_Jacobian_columns, h_J_max!, h_J_max_loc, z_scenario_loc, indices_h_scenario, indices_U, indices_X, indices_Y, index_J_max, n_u, n_x, n_y, H, K)

        # Function eval_J_u is empty if epigraph constraints are used.
        eval_J_u = nothing
    else
        # J_u is used as cost function instead of epigraph notation.
        nzrange_h_J_max = nothing
        cache_h_J_max = nothing

        # The following function evaluates the cost function J(U) for the input vector U_vec (U_vec = vec(U)).
        function eval_J_u(U_vec::AbstractVector)
            U = @views reshape(U_vec, n_u, H)
            return J(U)
        end
    end

    if build_hessian
        # Build the Hessian of the Lagrangian.
        # Note that each scenario k has its own state (x(1:H)^[k]) and output variables (y(1:H)^[k]), but all
        # scenarios share the same control‑inputs u(1:H). Consequently, the Hessians of the local Lagrangians
        # corresponding to the dynamic and scenario constraints may have entries in the (u,u) block of the global Hessian of the Lagrangian.
        # The entries in the (u,u) block of the global Hessian may therefore repeat across scenarios.
        # This may lead to duplicate entries in the global sparsity pattern of the Hessian of the Lagrangian.
        # This is indeed desired. The the MathOptInterface (MOI) will handle the duplicate entries by summing them up.
        # This keeps the threaded implementation simple and still exploits scenario–level parallelism efficiently.

        # Initialize sparsity pattern for the Hessian of the Lagrangian.
        sparsity_Hessian_Lagrangian_rows = Int32[] # global row indices of non-zero entries
        sparsity_Hessian_Lagrangian_columns = Int32[] # global column indices of non-zero entries

        # Compute the Hessian of the local Lagrangian of the dynamic constraints.
        lambda_h_dynamics_x = ones(n_x * (H - 1))
        lambda_h_dynamics_y = ones(n_y * H)

        # Compute the sparsity pattern of the Hessian of the local Lagrangian containing the dynamic constraints for a single scenario, replicate it for all scenarios, and build per-thread cache for the automatic differentiation.
        L_dynamics_x_example = (z_scenario) -> lagrangian_dynamics_x(lambda_h_dynamics_x, z_scenario, PMCMC_samples[1].theta, V[:, :, 1])
        L_dynamics_y_example = (z_scenario) -> lagrangian_dynamics_y(lambda_h_dynamics_y, z_scenario, PMCMC_samples[1].theta, W[:, :, 1])

        nzrange_Hessian_L_dynamics_x, cache_Hessian_L_dynamics_x = setup_sparse_Hessian_cache!(sparsity_Hessian_Lagrangian_rows, sparsity_Hessian_Lagrangian_columns, L_dynamics_x_example, z_scenario_loc, indices_h_dynamics_x, indices_U, indices_X, indices_Y, index_J_max, n_u, n_x, n_y, H, K)
        nzrange_Hessian_L_dynamics_y, cache_Hessian_L_dynamics_y = setup_sparse_Hessian_cache!(sparsity_Hessian_Lagrangian_rows, sparsity_Hessian_Lagrangian_columns, L_dynamics_y_example, z_scenario_loc, indices_h_dynamics_y, indices_U, indices_X, indices_Y, index_J_max, n_u, n_x, n_y, H, K)

        # Compute the sparsity pattern of the Hessian of the local Lagrangian containing the scenario constraints for a single scenario, replicate it for all scenarios, and build per-thread cache for the automatic differentiation.
        lambda_h_scenario = ones(n_h_scenario)
        L_scenario_example = (z_scenario) -> lagrangian_scenario(lambda_h_scenario, z_scenario)
        nzrange_Hessian_L_scenario, cache_Hessian_L_scenario = setup_sparse_Hessian_cache!(sparsity_Hessian_Lagrangian_rows, sparsity_Hessian_Lagrangian_columns, L_scenario_example, z_scenario_loc, indices_h_scenario, indices_U, indices_X, indices_Y, index_J_max, n_u, n_x, n_y, H, K)

        # Evaluate the sparsity pattern of the Hessian of the local Lagrangian containing the input constraints h_u.
        lambda_h_u = ones(n_h_u)
        lagrangian_u_example = (U_vec) -> lagrangian_u(lambda_h_u, U_vec)
        sparsity_Hessian_L_u, colors_Hessian_L_u, rows_Hessian_L_u, columns_Hessian_L_u = compute_Hessian_sparsity(lagrangian_u_example, u_loc)

        # Add the non-zero entries of the sparsity pattern of the Hessian of the local Lagrangian to the global sparsity pattern.
        start = length(sparsity_Hessian_Lagrangian_rows) + 1
        for (r, c) in zip(rows_Hessian_L_u, columns_Hessian_L_u)
            push!(sparsity_Hessian_Lagrangian_rows, indices_U[r])
            push!(sparsity_Hessian_Lagrangian_columns, indices_U[c])
        end
        stop = length(sparsity_Hessian_Lagrangian_rows)
        nzrange_Hessian_L_u = start:stop

        # Build cache for the automatic differentiation of L_u.
        cache_Hessian_L_u = ADCache(SparseDiffTools.ForwardColorHesCache(lagrangian_u_example, u_loc, colors_Hessian_L_u, sparsity_Hessian_L_u), Float64.(sparsity_Hessian_L_u))

        if !J_u
            # Compute the sparsity pattern of the Hessian of the local Lagrangian containing the epigraph constraints for a single scenario, replicate it for all scenarios, and build per-thread cache for the automatic differentiation.
            lambda_h_J_max = ones(1)
            L_J_max_example = (z_scenario) -> lagrangian_J_max(lambda_h_J_max, z_scenario)
            nzrange_Hessian_L_J_max, cache_Hessian_L_J_max = setup_sparse_Hessian_cache!(sparsity_Hessian_Lagrangian_rows, sparsity_Hessian_Lagrangian_columns, L_J_max_example, z_scenario, indices_h_J_max, indices_U, indices_X, indices_Y, index_J_max, n_u, n_x, n_y, H, K)

            # Function eval_J_u is empty if epigraph constraints are used.
            nzrange_Hessian_J_u = nothing
            cache_Hessian_J_u = nothing
        else
            # Compute the Hessian of the cost function J(u).
            Hessian_J_u_sparsity, Hessian_J_u_colors, rows_Hessian_J_u, columns_Hessian_J_u = compute_Hessian_sparsity(eval_J_u, u_loc)

            start = length(sparsity_Hessian_Lagrangian_rows) + 1
            for (r, c) in zip(rows_Hessian_J_u, columns_Hessian_J_u)
                push!(sparsity_Hessian_Lagrangian_rows, indices_U[r])
                push!(sparsity_Hessian_Lagrangian_columns, indices_U[c])
            end
            stop = length(sparsity_Hessian_Lagrangian_rows)
            nzrange_Hessian_J_u = start:stop

            # Build cache for the automatic differentiation of J_u.
            cache_Hessian_J_u = ADCache(SparseDiffTools.ForwardColorHesCache(eval_J_u, u_loc, Hessian_J_u_colors, Hessian_J_u_sparsity), Float64.(sparsity_Hessian_J_u))

            # Hessian of the Lagrangian is not built for the epigraph constraints.
            nzrange_Hessian_L_J_max = nothing
            cache_Hessian_L_J_max = nothing
        end

        # Create the caches for the automatic differentiation of the constraints for each thread.
        thread_cache = Vector{ThreadCache}(undef, n_threads)
        for i in 1:n_threads
            thread_cache[i] = ThreadCache(cache_h_dynamics_x, cache_h_dynamics_y, cache_h_scenario, cache_h_J_max, cache_Hessian_L_dynamics_x, cache_Hessian_L_dynamics_y, cache_Hessian_L_scenario, cache_Hessian_L_J_max)
        end

        # Convert the sparsity pattern represented by two vectors containing the row and column indices to a vector of tuples.
        Hessian_pattern_L = Vector{Tuple{Int,Int}}(undef, length(sparsity_Hessian_Lagrangian_rows))
        for i in eachindex(sparsity_Hessian_Lagrangian_rows)
            Hessian_pattern_L[i] = (sparsity_Hessian_Lagrangian_rows[i], sparsity_Hessian_Lagrangian_columns[i])
        end
    else
        # Hessian of the Lagrangian is not built.
        Hessian_pattern_L = nothing
        nzrange_Hessian_L_dynamics_x = nothing
        nzrange_Hessian_L_dynamics_y = nothing
        nzrange_Hessian_L_scenario = nothing
        nzrange_Hessian_L_J_max = nothing
        cache_Hessian_L_dynamics_x = nothing
        cache_Hessian_L_dynamics_y = nothing
        cache_Hessian_L_scenario = nothing
        cache_Hessian_L_J_max = nothing
        cache_Hessian_L_u = nothing
        cache_Hessian_J_u = nothing

        # Create the caches for the automatic differentiation of the constraints for each thread.
        thread_cache = Vector{ThreadCache}(undef, n_threads)
        for i in 1:n_threads
            thread_cache[i] = ThreadCache(cache_h_dynamics_x, cache_h_dynamics_y, cache_h_scenario, cache_h_J_max, nothing, nothing, nothing, nothing)
        end
    end

    # Quick check to ensure the sparsity patterns are valid.
    @assert maximum(sparsity_Jacobian_rows) <= n_h
    @assert maximum(sparsity_Jacobian_columns) <= n_z
    @assert length(sparsity_Jacobian_rows) == length(sparsity_Jacobian_columns)

    @assert maximum(sparsity_Hessian_Lagrangian_rows) <= n_z
    @assert maximum(sparsity_Hessian_Lagrangian_columns) <= n_z
    @assert length(sparsity_Hessian_Lagrangian_rows) == length(sparsity_Hessian_Lagrangian_columns)

    # Convert the sparsity pattern represented by two vectors containing the row and column indices to a vector of tuples.
    Jacobian_pattern_h = Vector{Tuple{Int,Int}}(undef, length(sparsity_Jacobian_rows))
    for i in eachindex(sparsity_Jacobian_rows)
        Jacobian_pattern_h[i] = (sparsity_Jacobian_rows[i], sparsity_Jacobian_columns[i])
    end

    # Create the evaluator.
    return PMCMC_OCP_Evaluator(K, H, n_u, n_x, n_y,
        n_z, n_h,
        PMCMC_samples, V, W, X_t,
        h_dynamics_x!, h_dynamics_y!, h_scenario!, h_u!, h_J_max!, eval_J_u, J_u,
        indices_U, indices_X, indices_Y, index_J_max,
        indices_h_dynamics_x, indices_h_dynamics_y, indices_h_scenario, indices_h_u, indices_h_J_max,
        z_sets, h_bounds,
        Jacobian_pattern_h,
        nzrange_h_dynamics_x, nzrange_h_dynamics_y, nzrange_h_scenario, nzrange_h_u, nzrange_h_J_max,
        thread_cache, cache_Jacobian_h_u)
end