# This file contains the constructor for a custom evaluator for a scenario-based optimal control problem.
# By exploiting the specific structure of the problem and the sparsity pattern, the evaluator can efficiently compute the objective, the objective gradient, the constraints and the constraint Jacobian.
# The evaluator utilizes thread parallelism to speed up the evaluation of the constraints and their Jacobian.
# To profit from the implementation ensure that multiple threads are available by setting the JULIA_NUM_THREADS environment variable.

# Evaluator struct.
struct PMCMC_OCP_Evaluator <: MOI.AbstractNLPEvaluator
    # Options
    options::OCPOptions

    # Dimensions.
    dimensions::OCPDimensions

    # Indices in the flat decision vector z and the constraint vector h(z).
    indices::OCPIndices

    # Data.
    data::OCPData

    # Bounds for the decision vector z.
    z_sets::Vector{MOI.AbstractScalarSet}

    # Bounds for the constraint vector h(z).
    h_bounds::Vector{MOI.NLPBoundsPair}

    # Global sparsity pattern of the constraint Jacobian.
    Jacobian_pattern_h::Vector{Tuple{Int,Int}}

    # Global sparsity pattern of the Hessian of the Lagrangian.
    Hessian_pattern_L::Union{Vector{Tuple{Int,Int}},Nothing}

    # Ranges of the constraints in the vector containing the non-zero entries of the global Jacobian.
    nzrange_Jacobian_h::SparseJacobianNZRanges

    # Ranges of the Hessian of the local Lagrangian in the vector containing the non-zero entries of the global Hessian.
    nzrange_Hessian_L::Union{SparseHessianNZRanges,Nothing}

    # Cache for the automatic differentiation.
    thread_cache::Vector{ThreadCache}
    global_cache::GlobalCache
end

# Constructor for PMCMC_OCP_Evaluator.
function PMCMC_OCP_Evaluator(PMCMC_samples::Vector{PMCMC_sample}, V::Array{Float64}, W::Array{Float64}, X_t::Array{Float64}, f_theta::Function, g_theta::Function, J::Function, J_u::Bool, h_scenario::Function, h_u::Function, n_u::Int, n_x::Int, n_y::Int, H::Int; build_Hessian::Bool=true, deduplicate_Hessian::Bool=false)
    # Check if multithreading is enabled.
    K = Threads.nthreads()

    @info "Evaluator running with $K Julia thread$(K == 1 ? "" : "s")."

    if K == 1
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

    # Create the options struct.
    options = OCPOptions(build_Hessian, deduplicate_Hessian, J_u)

    # Create the dimensions struct.
    dimensions = OCPDimensions(n_u, n_x, n_y, K, H, n_z, n_z_scenario, n_h_scenario, n_h_u, n_h)

    # Get the index ranges in the flat decision vector z corresponding to the inputs U, states X, outputs Y, and (optionally) J_max 
    # and the index ranges in the flat constraint vector h(z) corresponding to the dynamic constraints for the states,
    # the dynamic constraints for the outputs, scenario constraints h_scenario, the input constraints h_u, and (optionally) the epigraph constraints.
    indices = get_indices(options, dimensions)

    # Create the data struct.
    data = OCPData(PMCMC_samples, V, W, X_t)

    # Create thread and global context.
    # Lagrange multipliers
    lambda_h_dynamics_x = ones(n_x * (H - 1))
    lambda_h_dynamics_y = ones(n_y * H)
    lambda_h_scenario = ones(n_h_scenario)
    if !J_u
        lambda_h_J_max = ones(1)
    else
        lambda_h_J_max = nothing
    end

    thread_context = Vector{ThreadContext}(undef, K)
    for i in 1:K
        thread_context[i] = ThreadContext(copy(data.PMCMC_samples[i].theta), copy(data.V[:, :, i]), copy(data.W[:, :, i]), copy(lambda_h_dynamics_x), copy(lambda_h_dynamics_y), copy(lambda_h_scenario), copy(lambda_h_J_max))
    end

    lambda_h_u = ones(n_h_u)
    global_context = GlobalContext(lambda_h_u)

    # Build the helper functions that evaluate the scenario dependent constraints and their Lagrangians.
    thread_helpers = Vector{ThreadHelpers}(undef, K)
    for i in 1:K
        thread_helpers[i] = build_thread_helpers(f_theta, g_theta, h_scenario, J, options, dimensions, thread_context[i])
    end
    global_helpers = build_global_helpers(h_u, J, options, dimensions, global_context)

    # Create bounds for the flat decision vector z.
    z_sets = Vector{MOI.AbstractScalarSet}(undef, dimensions.n_z)
    for i in indices.U
        z_sets[i] = MOI.Interval(-Inf, Inf)
    end
    for k in 1:dimensions.K
        for (index_x, i) in enumerate(indices.X[k])
            if i <= dimensions.n_x
                # Initial state x_1^[k] is fixed.
                z_sets[indices.X[k][index_x]] = MOI.EqualTo(X_t[index_x, k])
            else
                z_sets[i] = MOI.Interval(-Inf, Inf)
            end
        end
        for i in indices.Y[k]
            z_sets[i] = MOI.Interval(-Inf, Inf)
        end
    end
    if !options.J_u
        for i in indices.J_max
            z_sets[i] = MOI.Interval(-Inf, Inf)
        end
    end

    # Get bounds for the constraint vector h(z).
    h_bounds = Vector{MOI.NLPBoundsPair}(undef, dimensions.n_h)
    for k in 1:dimensions.K
        # Dynamic (equality) constraints for the states.
        for i in indices.h_dynamics_x[k]
            h_bounds[i] = MOI.NLPBoundsPair(0.0, 0.0)
        end

        # Dynamic (equality) constraints for the outputs.
        for i in indices.h_dynamics_y[k]
            h_bounds[i] = MOI.NLPBoundsPair(0.0, 0.0)
        end

        # Scenario (inequality) constraints.
        for i in indices.h_scenario[k]
            h_bounds[i] = MOI.NLPBoundsPair(-Inf, 0.0)
        end

        # Epigraph (inequality) constraints.
        if !options.J_u
            for i in indices.h_J_max[k]
                h_bounds[i] = MOI.NLPBoundsPair(-Inf, 0.0)
            end
        end
    end
    # Input (inequality) constraints.
    for i in indices.h_u
        h_bounds[i] = MOI.NLPBoundsPair(-Inf, 0.0)
    end

    # Initialize sparsity pattern for the constraint Jacobian.
    sparsity_Jacobian_rows = Int[] # global row indices of non-zero entries
    sparsity_Jacobian_columns = Int[] # global column indices of non-zero entries

    # Compute the sparsity pattern for the Jacobian of the dynamic constraints.
    # The following vectors determine in which region the sparsity pattern is evaluated.
    # Without input dependent branches (e.g., min, max, if) the expression tree is fixed and the sparsity pattern does not depend on the actual values of these vectors.
    z_scenario_loc = zeros(dimensions.n_z_scenario)
    h_dynamics_x_loc = zeros(dimensions.n_x * (dimensions.H - 1))
    h_dynamics_y_loc = zeros(dimensions.n_y * dimensions.H)

    # Compute the sparsity pattern of the Jacobian of the dynamic constraints for a single scenario and replicate it for all scenarios.
    nzrange_h_dynamics_x, colors_h_dynamics_x, sparsity_h_dynamics_x = register_local_Jacobian_sparsity!(sparsity_Jacobian_rows, sparsity_Jacobian_columns, thread_helpers[1].h_dynamics_x!, h_dynamics_x_loc, z_scenario_loc, indices.h_dynamics_x, options, dimensions, indices)
    nzrange_h_dynamics_y, colors_h_dynamics_y, sparsity_h_dynamics_y = register_local_Jacobian_sparsity!(sparsity_Jacobian_rows, sparsity_Jacobian_columns, thread_helpers[1].h_dynamics_y!, h_dynamics_y_loc, z_scenario_loc, indices.h_dynamics_y, options, dimensions, indices)

    # Compute the sparsity pattern of the Jacobian of the scenario constraints for a single scenario and replicate it for all scenarios.
    h_scenario_loc = zeros(dimensions.n_h_scenario)
    nzrange_h_scenario, colors_h_scenario, sparsity_h_scenario = register_local_Jacobian_sparsity!(sparsity_Jacobian_rows, sparsity_Jacobian_columns, thread_helpers[1].h_scenario!, h_scenario_loc, z_scenario_loc, indices.h_scenario, options, dimensions, indices)

    # Evaluate the sparsity pattern of h_u.
    u_loc = zeros(dimensions.n_u * dimensions.H)
    h_u_loc = zeros(dimensions.n_h_u)
    Jac_pattern_h_u, colors_h_u, rows_h_u_local, columns_h_u_local = compute_Jacobian_sparsity(global_helpers.h_u!, u_loc, h_u_loc)

    # Add the non-zero entries of the sparsity pattern of h_u to the global sparsity pattern.
    row_offset = first(indices.h_u) - 1
    start = length(sparsity_Jacobian_rows) + 1
    for (r, c) in zip(rows_h_u_local, columns_h_u_local)
        push!(sparsity_Jacobian_rows, row_offset + r)
        push!(sparsity_Jacobian_columns, indices.U[c])
    end
    stop = length(sparsity_Jacobian_rows)
    nzrange_h_u = start:stop

    # Build cache for the automatic differentiation of h_u.
    cache_Jacobian_h_u = ADCache(SparseDiffTools.ForwardColorJacCache(global_helpers.h_u!, u_loc, nothing; dx=h_u_loc, colorvec=colors_h_u, sparsity=Jac_pattern_h_u), Float64.(Jac_pattern_h_u))

    if !options.J_u
        # Epigraph constraints are used.
        # Compute the sparsity pattern of the Jacobian of the epigraph constraint for a single scenario, replicate it for all scenarios, and build per-thread cache for the automatic differentiation.
        # The epigraph constraint for one scenario is scalar, so its Jacobian is a single row (i.e., the gradient).  With only one row, matrix colouring offers no benefit: we could build the sparsity pattern with
        # `Symbolics.jacobian_sparsity` once and then evaluate the row via `ForwardDiff.gradient!` in the callback.
        # Instead, we deliberately reuse the same Jacobian‑building path we use for the larger constraint blocks.  
        # This keeps the implementation uniform and the callback logic simple, at the cost of a negligible amount of extra work for this one row.
        h_J_max_loc = zeros(1)
        nzrange_h_J_max, colors_h_J_max, sparsity_h_J_max = register_local_Jacobian_sparsity!(sparsity_Jacobian_rows, sparsity_Jacobian_columns, thread_helpers[1].h_J_max!, h_J_max_loc, z_scenario_loc, indices.h_J_max, options, dimensions, indices)
    else
        # J_u is used as cost function instead of epigraph notation.
        nzrange_h_J_max = nothing
        colors_h_J_max = nothing
        sparsity_h_J_max = nothing
    end

    # Convert the sparsity pattern represented by two vectors containing the row and column indices to a vector of tuples.
    Jacobian_pattern_h = Vector{Tuple{Int,Int}}(undef, length(sparsity_Jacobian_rows))
    for i in eachindex(sparsity_Jacobian_rows)
        Jacobian_pattern_h[i] = (sparsity_Jacobian_rows[i], sparsity_Jacobian_columns[i])
    end

    # Create struct that contains the ranges corresponding to specific constraints in the vector containing the non-zero entries of the global Jacobian.
    nzrange_Jacobian_h = SparseJacobianNZRanges(nzrange_h_dynamics_x, nzrange_h_dynamics_y, nzrange_h_scenario, nzrange_h_u, nzrange_h_J_max)

    if build_Hessian
        # Build the Hessian of the Lagrangian.
        # Note that each scenario k has its own state (x(1:H)^[k]) and output variables (y(1:H)^[k]), but all
        # scenarios share the same control‑inputs u(1:H). Consequently, the Hessians of the local Lagrangians
        # corresponding to the dynamic and scenario constraints may have entries in the (u,u) block of the global Hessian of the Lagrangian.
        # The entries in the (u,u) block of the global Hessian may therefore repeat across scenarios.
        # This may lead to duplicate entries in the global sparsity pattern of the Hessian of the Lagrangian.
        # This is indeed desired. The the MathOptInterface (MOI) will handle the duplicate entries by summing them up.
        # This keeps the threaded implementation simple and still exploits scenario–level parallelism efficiently.

        # Initialize sparsity pattern for the Hessian of the Lagrangian.
        sparsity_Hessian_Lagrangian_rows = Int[] # global row indices of non-zero entries
        sparsity_Hessian_Lagrangian_columns = Int[] # global column indices of non-zero entries

        nzrange_Hessian_L_dynamics_x, colors_Hessian_L_dynamics_x, sparsity_Hessian_L_dynamics_x = register_local_Hessian_sparsity!(sparsity_Hessian_Lagrangian_rows, sparsity_Hessian_Lagrangian_columns, thread_helpers[1].lagrangian_dynamics_x, z_scenario_loc, options, dimensions, indices)
        nzrange_Hessian_L_dynamics_y, colors_Hessian_L_dynamics_y, sparsity_Hessian_L_dynamics_y = register_local_Hessian_sparsity!(sparsity_Hessian_Lagrangian_rows, sparsity_Hessian_Lagrangian_columns, thread_helpers[1].lagrangian_dynamics_y, z_scenario_loc, options, dimensions, indices)

        # Compute the sparsity pattern of the Hessian of the local Lagrangian containing the scenario constraints for a single scenario, replicate it for all scenarios, and build per-thread cache for the automatic differentiation.
        nzrange_Hessian_L_scenario, colors_Hessian_L_scenario, sparsity_Hessian_L_scenario = register_local_Hessian_sparsity!(sparsity_Hessian_Lagrangian_rows, sparsity_Hessian_Lagrangian_columns, thread_helpers[1].lagrangian_scenario, z_scenario_loc, options, dimensions, indices)

        # Evaluate the sparsity pattern of the Hessian of the local Lagrangian containing the input constraints h_u.
        sparsity_Hessian_L_u, colors_Hessian_L_u, rows_Hessian_L_u, columns_Hessian_L_u = compute_Hessian_sparsity(global_helpers.lagrangian_u, u_loc)

        # Add the non-zero entries of the sparsity pattern of the Hessian of the local Lagrangian to the global sparsity pattern.
        start = length(sparsity_Hessian_Lagrangian_rows) + 1
        for (r, c) in zip(rows_Hessian_L_u, columns_Hessian_L_u)
            push!(sparsity_Hessian_Lagrangian_rows, indices.U[r])
            push!(sparsity_Hessian_Lagrangian_columns, indices.U[c])
        end
        stop = length(sparsity_Hessian_Lagrangian_rows)
        nzrange_Hessian_L_u = start:stop

        # Build cache for the automatic differentiation of L_u.
        cache_Hessian_L_u = ADCache(SparseDiffTools.ForwardAutoColorHesCache(global_helpers.lagrangian_u, u_loc, colors_Hessian_L_u, sparsity_Hessian_L_u), Float64.(sparsity_Hessian_L_u))

        if !options.J_u
            # Compute the sparsity pattern of the Hessian of the local Lagrangian containing the epigraph constraints for a single scenario, replicate it for all scenarios, and build per-thread cache for the automatic differentiation.
            nzrange_Hessian_L_J_max, colors_Hessian_L_J_max, sparsity_Hessian_L_J_max = register_local_Hessian_sparsity!(sparsity_Hessian_Lagrangian_rows, sparsity_Hessian_Lagrangian_columns, thread_helpers[1].lagrangian_J_max, z_scenario_loc, options, dimensions, indices)

            # Function eval_J_u is empty if epigraph constraints are used.
            nzrange_Hessian_J_u = nothing
            cache_Hessian_J_u = nothing
        else
            # Compute the Hessian of the cost function J(u).
            Hessian_J_u_sparsity, Hessian_J_u_colors, rows_Hessian_J_u, columns_Hessian_J_u = compute_Hessian_sparsity(global_helpers.eval_J_u, u_loc)

            start = length(sparsity_Hessian_Lagrangian_rows) + 1
            for (r, c) in zip(rows_Hessian_J_u, columns_Hessian_J_u)
                push!(sparsity_Hessian_Lagrangian_rows, indices.U[r])
                push!(sparsity_Hessian_Lagrangian_columns, indices.U[c])
            end
            stop = length(sparsity_Hessian_Lagrangian_rows)
            nzrange_Hessian_J_u = start:stop

            # Build cache for the automatic differentiation of J_u.
            cache_Hessian_J_u = ADCache(SparseDiffTools.ForwardAutoColorHesCache(global_helpers.eval_J_u, u_loc, Hessian_J_u_colors, Hessian_J_u_sparsity), Float64.(sparsity_Hessian_J_u))

            # Hessian of the Lagrangian is not built for the epigraph constraints.
            nzrange_Hessian_L_J_max = nothing
        end

        # Convert the sparsity pattern represented by two vectors containing the row and column indices to a vector of tuples.
        Hessian_pattern_L = Vector{Tuple{Int,Int}}(undef, length(sparsity_Hessian_Lagrangian_rows))
        for i in eachindex(sparsity_Hessian_Lagrangian_rows)
            Hessian_pattern_L[i] = (sparsity_Hessian_Lagrangian_rows[i], sparsity_Hessian_Lagrangian_columns[i])
        end

        # Create the struct that contains the ranges corresponding to specific constraints in the vector containing the non-zero entries of the global Hessian.
        nzrange_Hessian_L = SparseHessianNZRanges(nzrange_Hessian_L_dynamics_x, nzrange_Hessian_L_dynamics_y, nzrange_Hessian_L_scenario, nzrange_Hessian_L_u, nzrange_Hessian_L_J_max, nzrange_Hessian_J_u)

    else
        # Hessian of the Lagrangian is not built.
        Hessian_pattern_L = nothing
        nzrange_Hessian_L = nothing
        cache_Hessian_L_u = nothing
        cache_Hessian_J_u = nothing
    end

    # Create the thread cache for the automatic differentiation of the dynamic, scenario, and epigraph constraints and their Lagrangians.
    thread_cache = Vector{ThreadCache}(undef, K)
    for i in 1:K
        cache_h_dynamics_x = ADCache(SparseDiffTools.ForwardColorJacCache(thread_helpers[i].h_dynamics_x!, copy(z_scenario_loc), nothing; dx=h_dynamics_x_loc, colorvec=copy(colors_h_dynamics_x), sparsity=copy(sparsity_h_dynamics_x)), copy(Float64.(sparsity_h_dynamics_x)))
        cache_h_dynamics_y = ADCache(SparseDiffTools.ForwardColorJacCache(thread_helpers[i].h_dynamics_y!, copy(z_scenario_loc), nothing; dx=h_dynamics_y_loc, colorvec=copy(colors_h_dynamics_y), sparsity=copy(sparsity_h_dynamics_y)), copy(Float64.(sparsity_h_dynamics_y)))
        cache_h_scenario = ADCache(SparseDiffTools.ForwardColorJacCache(thread_helpers[i].h_scenario!, copy(z_scenario_loc), nothing; dx=h_scenario_loc, colorvec=copy(colors_h_scenario), sparsity=copy(sparsity_h_scenario)), copy(Float64.(sparsity_h_scenario)))
        if !options.J_u
            cache_h_J_max = ADCache(SparseDiffTools.ForwardColorJacCache(thread_helpers[i].h_J_max!, copy(z_scenario_loc), nothing; dx=h_J_max_loc, colorvec=copy(colors_h_J_max), sparsity=copy(sparsity_h_J_max)), copy(Float64.(sparsity_h_J_max)))
        else
            cache_h_J_max = nothing
        end
        if build_Hessian
            cache_Hessian_L_dynamics_x = ADCache(SparseDiffTools.ForwardAutoColorHesCache(thread_helpers[i].lagrangian_dynamics_x, copy(z_scenario_loc), copy(colors_Hessian_L_dynamics_x), copy(sparsity_Hessian_L_dynamics_x)), copy(Float64.(sparsity_Hessian_L_dynamics_x)))
            cache_Hessian_L_dynamics_y = ADCache(SparseDiffTools.ForwardAutoColorHesCache(thread_helpers[i].lagrangian_dynamics_y, copy(z_scenario_loc), copy(colors_Hessian_L_dynamics_y), copy(sparsity_Hessian_L_dynamics_y)), copy(Float64.(sparsity_Hessian_L_dynamics_y)))
            cache_Hessian_L_scenario = ADCache(SparseDiffTools.ForwardAutoColorHesCache(thread_helpers[i].lagrangian_scenario, copy(z_scenario_loc), copy(colors_Hessian_L_scenario), copy(sparsity_Hessian_L_scenario)), copy(Float64.(sparsity_Hessian_L_scenario)))
            if !options.J_u
                cache_Hessian_L_J_max = ADCache(SparseDiffTools.ForwardAutoColorHesCache(thread_helpers[i].lagrangian_J_max, copy(z_scenario_loc), copy(colors_Hessian_L_J_max), copy(sparsity_Hessian_L_J_max)), copy(Float64.(sparsity_Hessian_L_J_max)))
            else
                cache_Hessian_L_J_max = nothing
            end
        else
            cache_Hessian_L_dynamics_x = nothing
            cache_Hessian_L_dynamics_y = nothing
            cache_Hessian_L_scenario = nothing
            cache_Hessian_L_J_max = nothing
        end
        thread_cache[i] = ThreadCache(cache_h_dynamics_x, cache_h_dynamics_y, cache_h_scenario, cache_h_J_max, cache_Hessian_L_dynamics_x, cache_Hessian_L_dynamics_y, cache_Hessian_L_scenario, cache_Hessian_L_J_max, thread_helpers[i], thread_context[i])
    end

    # Create the global cache for the automatic differentiation of the constraints.
    global_cache = GlobalCache(cache_Jacobian_h_u, cache_Hessian_L_u, cache_Hessian_J_u, global_context, global_helpers)

    # Quick check to ensure the sparsity patterns are valid.
    @assert maximum(sparsity_Jacobian_rows) <= n_h
    @assert maximum(sparsity_Jacobian_columns) <= n_z
    @assert length(sparsity_Jacobian_rows) == length(sparsity_Jacobian_columns)

    if build_Hessian
        @assert maximum(sparsity_Hessian_Lagrangian_rows) <= n_z
        @assert maximum(sparsity_Hessian_Lagrangian_columns) <= n_z
        @assert length(sparsity_Hessian_Lagrangian_rows) == length(sparsity_Hessian_Lagrangian_columns)
    end

    # Create the evaluator.
    return PMCMC_OCP_Evaluator(options,
        dimensions,
        indices,
        data,
        z_sets,
        h_bounds,
        Jacobian_pattern_h,
        Hessian_pattern_L,
        nzrange_Jacobian_h,
        nzrange_Hessian_L,
        thread_cache,
        global_cache)
end