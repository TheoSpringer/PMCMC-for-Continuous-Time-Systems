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

    # Functions.
    helpers::OCPFunctions

    # Bounds for the decision vector z.
    z_sets::Vector{MOI.AbstractScalarSet}

    # Bounds for the constraint vector h(z).
    h_bounds::Vector{MOI.NLPBoundsPair}

    # Global sparsity pattern of the constraint Jacobian.
    sparsity_Jac_h::Vector{Tuple{Int,Int}}

    # Global sparsity pattern of the Hessian of the Lagrangian.
    sparsity_Hes_L::Union{Vector{Tuple{Int,Int}},Nothing}

    # Ranges of the constraints in the vector containing the non-zero entries of the global Jacobian.
    nzrange_Jac_h::SparseJacobianNZRanges

    # Ranges of the Hessian of the local Lagrangian in the vector containing the non-zero entries of the global Hessian.
    nzrange_Hes_L::Union{SparseHessianNZRanges,Nothing}

    # Backends and cache for the automatic differentiation.
    backends::ADBackends
    thread_cache::Vector{ThreadCache}
    global_cache::GlobalCache
end

# Constructor for PMCMC_OCP_Evaluator.
function PMCMC_OCP_Evaluator(PMCMC_samples::Vector{PMCMC_sample}, V::Array{Float64}, W::Array{Float64}, X_t::Array{Float64}, f_theta::Function, g_theta::Function, J::Function, J_u::Bool, h_scenario::Function, h_u::Function, n_u::Int, n_x::Int, n_y::Int, H::Int; build_Hessian::Bool=true, deduplicate_Hessian::Bool=false, sparsity_detector::AbstractSparsityDetector=TracerSparsityDetector(), coloring_algorithm::AbstractColoringAlgorithm=GreedyColoringAlgorithm(), dense_forward_backend=AutoForwardDiff(), dense_second_order_backend=SecondOrder(AutoForwardDiff(), ADTypes.AutoReverseDiff()))
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
    options = OCPOptions(build_Hessian, deduplicate_Hessian, J_u, sparsity_detector, coloring_algorithm, dense_forward_backend, dense_second_order_backend)

    # Create the dimensions struct.
    dimensions = OCPDimensions(n_u, n_x, n_y, K, H, n_z, n_z_scenario, n_h_scenario, n_h_u, n_h)

    # Get the index ranges in the flat decision vector z corresponding to the inputs U, states X, outputs Y, and (optionally) J_max 
    # and the index ranges in the flat constraint vector h(z) corresponding to the dynamic constraints for the states,
    # the dynamic constraints for the outputs, scenario constraints h_scenario, the input constraints h_u, and (optionally) the epigraph constraints.
    indices = get_indices(options, dimensions)

    # Create the data struct.
    data = OCPData(PMCMC_samples, V, W, X_t)

    # Build the helper functions that evaluate the scenario dependent constraints and their Lagrangians.
    helpers = build_helpers(f_theta, g_theta, h_scenario, h_u, J, options, dimensions)

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
    sparsity_Jac_h_rows = Int[] # global row indices of non-zero entries
    sparsity_Jac_h_columns = Int[] # global column indices of non-zero entries

    # Compute the sparsity pattern for the Jacobian of the dynamic constraints.
    # The following vectors determine in which region the sparsity pattern is evaluated.
    # Without input dependent branches (e.g., min, max, if) the expression tree is fixed and the sparsity pattern does not depend on the actual values of these vectors.
    z_scenario_loc = zeros(dimensions.n_z_scenario)
    h_dynamics_x_loc = zeros(dimensions.n_x * (dimensions.H - 1))
    h_dynamics_y_loc = zeros(dimensions.n_y * dimensions.H)

    # Compute the sparsity pattern of the Jacobian of the dynamic constraints for a single scenario and replicate it for all scenarios.
    h_dynamics_x_example! = (h_dyn_x, z_scenario) -> helpers.h_dynamics_x!(h_dyn_x, z_scenario, data.PMCMC_samples[1].theta, data.V[:, :, 1])
    h_dynamics_y_example! = (h_dyn_y, z_scenario) -> helpers.h_dynamics_y!(h_dyn_y, z_scenario, data.PMCMC_samples[1].theta, data.W[:, :, 1])
    sparsity_Jac_h_dynamics_x, nzrange_Jac_h_dynamics_x, backend_Jac_h_dynamics_x = register_local_Jacobian_sparsity!(sparsity_Jac_h_rows, sparsity_Jac_h_columns, h_dynamics_x_example!, h_dynamics_x_loc, z_scenario_loc, indices.h_dynamics_x, options, dimensions, indices)
    sparsity_Jac_h_dynamics_y, nzrange_Jac_h_dynamics_y, backend_Jac_h_dynamics_y = register_local_Jacobian_sparsity!(sparsity_Jac_h_rows, sparsity_Jac_h_columns, h_dynamics_y_example!, h_dynamics_y_loc, z_scenario_loc, indices.h_dynamics_y, options, dimensions, indices)

    # Compute the sparsity pattern of the Jacobian of the scenario constraints for a single scenario and replicate it for all scenarios.
    h_scenario_loc = zeros(dimensions.n_h_scenario)
    sparsity_Jac_h_scenario, nzrange_Jac_h_scenario, backend_Jac_h_scenario = register_local_Jacobian_sparsity!(sparsity_Jac_h_rows, sparsity_Jac_h_columns, helpers.h_scenario!, h_scenario_loc, z_scenario_loc, indices.h_scenario, options, dimensions, indices)

    # Evaluate the sparsity pattern of h_u.
    u_loc = zeros(dimensions.n_u * dimensions.H)
    h_u_loc = zeros(dimensions.n_h_u)
    sparsity_Jac_h_u, coloring_Jac_h_u, rows_Jac_h_u_local, columns_Jac_h_u_local = compute_Jacobian_sparsity(helpers.h_u!, u_loc, h_u_loc, options)

    # Add the non-zero entries of the sparsity pattern of h_u to the global sparsity pattern.
    row_offset = first(indices.h_u) - 1
    start = length(sparsity_Jac_h_rows) + 1
    for (r, c) in zip(rows_Jac_h_u_local, columns_Jac_h_u_local)
        push!(sparsity_Jac_h_rows, row_offset + r)
        push!(sparsity_Jac_h_columns, indices.U[c])
    end
    stop = length(sparsity_Jac_h_rows)
    nzrange_Jac_h_u = start:stop

    # Build cache for the automatic differentiation of h_u.
    const_sparsity_detector_Jac_h_u = ADTypes.KnownJacobianSparsityDetector(sparsity_Jac_h_u)
    const_coloring_algorithm_Jac_h_u = SparseMatrixColorings.ConstantColoringAlgorithm(sparsity_Jac_h_u, coloring_Jac_h_u; partition=coloring_partition(options.dense_forward_backend))
    backend_Jac_h_u = DifferentiationInterface.AutoSparse(options.dense_forward_backend, const_sparsity_detector_Jac_h_u, const_coloring_algorithm_Jac_h_u)
    output_h_u = zeros(dimensions.n_h_u)
    u_view = @views u_loc[1:length(u_loc)]
    prep_Jac_h_u = DifferentiationInterface.prepare_jacobian(helpers.h_u!, output_h_u, backend_Jac_h_u, u_view)

    if !options.J_u
        # Epigraph constraints are used.
        # Compute the sparsity pattern of the Jacobian of the epigraph constraint for a single scenario, replicate it for all scenarios, and build per-thread cache for the automatic differentiation.
        # The epigraph constraint for one scenario is scalar, so its Jacobian is a single row (i.e., the gradient).  With only one row, matrix colouring offers no benefit: we could build the sparsity pattern with
        # once and then evaluate the row via `gradient!` in the callback.
        # Instead, we deliberately reuse the same Jacobian‑building path we use for the larger constraint blocks.
        # This keeps the implementation uniform and the callback logic simple, at the cost of a negligible amount of extra work for this one row.
        h_J_max_loc = zeros(1)
        sparsity_Jac_h_J_max, nzrange_Jac_h_J_max, backend_Jac_h_J_max = register_local_Jacobian_sparsity!(sparsity_Jac_h_rows, sparsity_Jac_h_columns, helpers.h_J_max!, h_J_max_loc, z_scenario_loc, indices.h_J_max, options, dimensions, indices)
    else
        # J_u is used as cost function instead of epigraph notation.
        sparsity_Jac_h_J_max = nothing
        nzrange_Jac_h_J_max = nothing
        backend_Jac_h_J_max = nothing
    end

    # Convert the sparsity pattern represented by two vectors containing the row and column indices to a vector of tuples.
    sparsity_Jac_h = Vector{Tuple{Int,Int}}(undef, length(sparsity_Jac_h_rows))
    for i in eachindex(sparsity_Jac_h_rows)
        sparsity_Jac_h[i] = (sparsity_Jac_h_rows[i], sparsity_Jac_h_columns[i])
    end

    # Create struct that contains the ranges corresponding to specific constraints in the vector containing the non-zero entries of the global Jacobian.
    nzrange_Jac_h = SparseJacobianNZRanges(nzrange_Jac_h_dynamics_x, nzrange_Jac_h_dynamics_y, nzrange_Jac_h_scenario, nzrange_Jac_h_u, nzrange_Jac_h_J_max)

    if build_Hessian
        # Build the Hessian of the Lagrangian.
        # Note that each scenario k has its own state (x(1:H)^[k]) and output variables (y(1:H)^[k]), but all
        # scenarios share the same control‑inputs u(1:H). Consequently, the Hessians of the local Lagrangians
        # corresponding to the dynamic and scenario constraints may have entries in the (u,u) block of the global Hessian of the Lagrangian.
        # The entries in the (u,u) block of the global Hessian may therefore repeat across scenarios.
        # This may lead to duplicate entries in the global sparsity pattern of the Hessian of the Lagrangian.
        # This is indeed desired. The the MathOptInterface (MOI) will handle the duplicate entries by summing them up.
        # This keeps the threaded implementation simple and still exploits scenario–level parallelism efficiently.
        # In case the duplicate entries are not desired, the `deduplicate_Hessian` option can be set to `true`.

        # Initialize sparsity pattern for the Hessian of the Lagrangian.
        sparsity_Hes_L_rows = Int[] # global row indices of non-zero entries
        sparsity_Hes_L_columns = Int[] # global column indices of non-zero entries

        # Compute the sparsity pattern of the Hessian of the local Lagrangian containing the dynamic constraints for a single scenario and replicate it for all scenarios.
        lambda_h_dynamics_x_example = ones(dimensions.n_x * (dimensions.H - 1))
        lambda_h_dynamics_y_example = ones(dimensions.n_y * dimensions.H)
        L_dynamics_x_example = (z_scenario) -> helpers.lagrangian_dynamics_x(z_scenario, lambda_h_dynamics_x_example, data.PMCMC_samples[1].theta, data.V[:, :, 1])
        L_dynamics_y_example = (z_scenario) -> helpers.lagrangian_dynamics_y(z_scenario, lambda_h_dynamics_y_example, data.PMCMC_samples[1].theta, data.W[:, :, 1])

        sparsity_Hes_L_dynamics_x, nzrange_Hes_L_dynamics_x, backend_Hes_L_dynamics_x = register_local_Hessian_sparsity!(sparsity_Hes_L_rows, sparsity_Hes_L_columns, L_dynamics_x_example, z_scenario_loc, options, dimensions, indices)
        sparsity_Hes_L_dynamics_y, nzrange_Hes_L_dynamics_y, backend_Hes_L_dynamics_y = register_local_Hessian_sparsity!(sparsity_Hes_L_rows, sparsity_Hes_L_columns, L_dynamics_y_example, z_scenario_loc, options, dimensions, indices)

        # Compute the sparsity pattern of the Hessian of the local Lagrangian containing the scenario constraints for a single scenario and replicate it for all scenarios.
        lambda_h_scenario_example = ones(dimensions.n_h_scenario)
        L_scenario_example = (z_scenario) -> helpers.lagrangian_scenario(z_scenario, lambda_h_scenario_example)

        sparsity_Hes_L_scenario, nzrange_Hes_L_scenario, backend_Hes_L_scenario = register_local_Hessian_sparsity!(sparsity_Hes_L_rows, sparsity_Hes_L_columns, L_scenario_example, z_scenario_loc, options, dimensions, indices)

        # Evaluate the sparsity pattern of the Hessian of the local Lagrangian containing the input constraints h_u.
        lambda_h_u_example = ones(dimensions.n_h_u)
        L_u_example = (U_vec) -> helpers.lagrangian_u(U_vec, lambda_h_u_example)

        sparsity_Hes_L_u, rows_Hes_L_u, columns_Hes_L_u = compute_Hessian_sparsity(L_u_example, u_loc, options)

        # Add the non-zero entries of the sparsity pattern of the Hessian of the local Lagrangian to the global sparsity pattern.
        start = length(sparsity_Hes_L_rows) + 1
        for (r, c) in zip(rows_Hes_L_u, columns_Hes_L_u)
            push!(sparsity_Hes_L_rows, indices.U[r])
            push!(sparsity_Hes_L_columns, indices.U[c])
        end
        stop = length(sparsity_Hes_L_rows)
        nzrange_Hes_L_u = start:stop

        # Build cache for the automatic differentiation of L_u.
        const_sparsity_detector_Hes_L_u = ADTypes.KnownHessianSparsityDetector(sparsity_Hes_L_u)
        const_coloring_algorithm_Hes_L_u = ConstantSymmetricColoringAlgorithm(sparsity_Hes_L_u, algorithm=options.coloring_algorithm)
        backend_Hes_L_u = DifferentiationInterface.AutoSparse(options.dense_second_order_backend, const_sparsity_detector_Hes_L_u, const_coloring_algorithm_Hes_L_u)
        lambda_h_u_view = @views lambda_h_u_example[1:length(lambda_h_u_example)]
        prep_Hes_L_u = DifferentiationInterface.prepare_hessian(helpers.lagrangian_u, backend_Hes_L_u, u_view, Constant(lambda_h_u_view))

        if !options.J_u
            # Compute the sparsity pattern of the Hessian of the local Lagrangian containing the epigraph constraints for a single scenario and replicate it for all scenarios.
            lambda_h_J_max_example = ones(1)
            L_J_max_example = (z_scenario) -> helpers.lagrangian_J_max(z_scenario, lambda_h_J_max_example)

            sparsity_Hes_L_J_max, nzrange_Hes_L_J_max, backend_Hes_L_J_max = register_local_Hessian_sparsity!(sparsity_Hes_L_rows, sparsity_Hes_L_columns, L_J_max_example, z_scenario_loc, options, dimensions, indices)

            # Function eval_J_u is empty if epigraph constraints are used.
            sparsity_Hes_J_u = nothing
            nzrange_Hes_J_u = nothing
            backend_Hes_J_u = nothing
            prep_Hes_J_u = nothing
        else
            # Compute the Hessian of the cost function J(u).
            sparsity_Hes_J_u, rows_Hes_J_u, columns_Hes_J_u = compute_Hessian_sparsity(helpers.eval_J_u, u_loc)

            start = length(sparsity_Hes_L_rows) + 1
            for (r, c) in zip(rows_Hes_J_u, columns_Hes_J_u)
                push!(sparsity_Hes_L_rows, indices.U[r])
                push!(sparsity_Hes_L_columns, indices.U[c])
            end
            stop = length(sparsity_Hes_L_rows)
            nzrange_Hes_J_u = start:stop

            # Build cache for the automatic differentiation of J_u.
            const_sparsity_detector_Hes_J_u = ADTypes.KnownJacobianSparsityDetector(sparsity_Hes_J_u)
            const_coloring_algorithm_Hes_J_u = ConstantSymmetricColoringAlgorithm(sparsity_Hes_J_u, algorithm=options.coloring_algorithm)
            backend_Hes_J_u = DifferentiationInterface.AutoSparse(options.dense_second_order_backend, const_sparsity_detector_Hes_J_u, const_coloring_algorithm_Hes_J_u)
            prep_Hes_J_u = DifferentiationInterface.prepare_hessian(helpers.eval_J_u, backend_Hes_J_u, u_loc)

            # Hessian of the Lagrangian is not built for the epigraph constraints.
            sparsity_Hes_L_J_max = nothing
            nzrange_Hes_L_J_max = nothing
            backend_Hes_L_J_max = nothing
        end

        # Convert the sparsity pattern represented by two vectors containing the row and column indices to a vector of tuples.
        sparsity_Hes_L = Vector{Tuple{Int,Int}}(undef, length(sparsity_Hes_L_rows))
        for i in eachindex(sparsity_Hes_L_rows)
            sparsity_Hes_L[i] = (sparsity_Hes_L_rows[i], sparsity_Hes_L_columns[i])
        end

        # Create the struct that contains the ranges corresponding to specific constraints in the vector containing the non-zero entries of the global Hessian.
        nzrange_Hes_L = SparseHessianNZRanges(nzrange_Hes_L_dynamics_x, nzrange_Hes_L_dynamics_y, nzrange_Hes_L_scenario, nzrange_Hes_L_u, nzrange_Hes_L_J_max, nzrange_Hes_J_u)

    else
        # Hessian of the Lagrangian is not built.
        sparsity_Hes_L = nothing
        nzrange_Hes_L = nothing

        prep_Hes_L_u = nothing
        prep_Hes_J_u = nothing

        sparsity_Hes_L_dynamics_x = nothing
        sparsity_Hes_L_dynamics_y = nothing
        sparsity_Hes_L_scenario = nothing
        sparsity_Hes_L_u = nothing
        sparsity_Hes_L_J_max = nothing
        sparsity_Hes_J_u = nothing

        backend_Hes_L_dynamics_x = nothing
        backend_Hes_L_dynamics_y = nothing
        backend_Hes_L_scenario = nothing
        backend_Hes_L_u = nothing
        backend_Hes_L_J_max = nothing
        backend_Hes_J_u = nothing
    end

    # Store the backends.
    backends = ADBackends(backend_Jac_h_dynamics_x, backend_Jac_h_dynamics_y, backend_Jac_h_scenario, backend_Jac_h_u, backend_Jac_h_J_max, backend_Hes_L_dynamics_x, backend_Hes_L_dynamics_y, backend_Hes_L_scenario, backend_Hes_L_u, backend_Hes_L_J_max, backend_Hes_J_u)

    # Create the thread cache for the automatic differentiation of the dynamic, scenario, and epigraph constraints and their Lagrangians.
    thread_cache = Vector{ThreadCache}(undef, n_threads)
    for i in 1:n_threads
        # Get local variables.
        z_scenario_view = @views z_scenario_loc
        theta_k = @views data.PMCMC_samples[1].theta
        V_k = @views data.V[:, :, 1]
        W_k = @views data.W[:, :, 1]
        lambda_h_dynamics_x_view = @views lambda_h_dynamics_x_example[1:length(lambda_h_dynamics_x_example)]
        lambda_h_dynamics_y_view = @views lambda_h_dynamics_y_example[1:length(lambda_h_dynamics_y_example)]
        lambda_h_scenario_view = @views lambda_h_scenario_example[1:length(lambda_h_scenario_example)]
        if !options.J_u
            lambda_h_J_max_view = @views lambda_h_J_max_example[1:length(lambda_h_J_max_example)]
        end

        # Jacobian templates
        Jac_h_dynamics_x = copy(sparsity_Jac_h_dynamics_x)
        Jac_h_dynamics_y = copy(sparsity_Jac_h_dynamics_y)
        Jac_h_scenario = copy(sparsity_Jac_h_scenario)
        if !options.J_u
            Jac_h_J_max = copy(sparsity_Jac_h_J_max)
        else
            Jac_h_J_max = nothing
        end

        # Hessian templates
        if options.build_Hessian
            Hes_L_dynamics_x = copy(sparsity_Hes_L_dynamics_x)
            Hes_L_dynamics_y = copy(sparsity_Hes_L_dynamics_y)
            Hes_L_scenario = copy(sparsity_Hes_L_scenario)
            if !options.J_u
                Hes_L_J_max = copy(sparsity_Hes_L_J_max)
            else
                Hes_L_J_max = nothing
            end
        else
            Hes_L_dynamics_x = nothing
            Hes_L_dynamics_y = nothing
            Hes_L_scenario = nothing
            Hes_L_J_max = nothing
        end

        # Output templates
        output_h_dynamics_x = copy(h_dynamics_x_loc)
        output_h_dynamics_y = copy(h_dynamics_y_loc)
        output_h_scenario = copy(h_scenario_loc)
        if !options.J_u
            output_h_J_max = copy(h_J_max_loc)
        else
            output_h_J_max = nothing
        end

        # Jacobian preparations
        prep_Jac_h_dynamics_x = DifferentiationInterface.prepare_jacobian(helpers.h_dynamics_x!, output_h_dynamics_x, backend_Jac_h_dynamics_x, z_scenario_view, Constant(theta_k), Constant(V_k))
        prep_Jac_h_dynamics_y = DifferentiationInterface.prepare_jacobian(helpers.h_dynamics_y!, output_h_dynamics_y, backend_Jac_h_dynamics_y, z_scenario_view, Constant(theta_k), Constant(W_k))
        prep_Jac_h_scenario = DifferentiationInterface.prepare_jacobian(helpers.h_scenario!, output_h_scenario, backend_Jac_h_scenario, z_scenario_view)

        if !options.J_u
            # Epigraph constraints are used.
            prep_Jac_h_J_max = DifferentiationInterface.prepare_jacobian(helpers.h_J_max!, output_h_J_max, backend_Jac_h_J_max, z_scenario_view)
        else
            # J_u is used as cost function instead of epigraph notation.
            prep_Jac_h_J_max = nothing
        end

        # Hessian preparations
        if options.build_Hessian
            prep_Hes_L_dynamics_x = DifferentiationInterface.prepare_hessian(helpers.lagrangian_dynamics_x, backend_Hes_L_dynamics_x, z_scenario_view, Constant(lambda_h_dynamics_x_view), Constant(theta_k), Constant(V_k))
            prep_Hes_L_dynamics_y = DifferentiationInterface.prepare_hessian(helpers.lagrangian_dynamics_y, backend_Hes_L_dynamics_y, z_scenario_view, Constant(lambda_h_dynamics_y_view), Constant(theta_k), Constant(W_k))
            prep_Hes_L_scenario = DifferentiationInterface.prepare_hessian(helpers.lagrangian_scenario, backend_Hes_L_scenario, z_scenario_view, Constant(lambda_h_scenario_view))

            if !options.J_u
                # Epigraph constraints are used.
                prep_Hes_L_J_max = DifferentiationInterface.prepare_hessian(helpers.lagrangian_J_max, backend_Hes_L_J_max, z_scenario_view, Constant(lambda_h_J_max_view))
            else
                # J_u is used as cost function instead of epigraph notation.
                prep_Hes_L_J_max = nothing
            end

        else
            # Hessian of the Lagrangian is not built.
            prep_Hes_L_dynamics_x = nothing
            prep_Hes_L_dynamics_y = nothing
            prep_Hes_L_scenario = nothing
            prep_Hes_L_J_max = nothing
        end

        thread_cache[i] = ThreadCache(
            # Jacobian preparations
            prep_Jac_h_dynamics_x,
            prep_Jac_h_dynamics_y,
            prep_Jac_h_scenario,
            prep_Jac_h_J_max,
            # Hessian preparations
            prep_Hes_L_dynamics_x,
            prep_Hes_L_dynamics_y,
            prep_Hes_L_scenario,
            prep_Hes_L_J_max,
            # Jacobian templates
            Jac_h_dynamics_x,
            Jac_h_dynamics_y,
            Jac_h_scenario,
            Jac_h_J_max,
            # Hessian templates
            Hes_L_dynamics_x,
            Hes_L_dynamics_y,
            Hes_L_scenario,
            Hes_L_J_max,
            # Output templates
            output_h_dynamics_x,
            output_h_dynamics_y,
            output_h_scenario,
            output_h_J_max)
    end

    # Create the global cache for the automatic differentiation of the constraints.
    Jac_h_u = copy(sparsity_Jac_h_u)
    if options.build_Hessian
        Hes_L_u = copy(sparsity_Hes_L_u)
        if !options.J_u
            Hes_J_u = nothing
        else
            Hes_J_u = copy(sparsity_Hes_J_u)
        end
    else
        Hes_L_u = nothing
        Hes_J_u = nothing
    end
    global_cache = GlobalCache(prep_Jac_h_u, prep_Hes_L_u, prep_Hes_J_u, Jac_h_u, Hes_L_u, Hes_J_u, output_h_u)

    # Quick check to ensure the sparsity patterns are valid.
    @assert maximum(sparsity_Jac_h_rows) <= n_h
    @assert maximum(sparsity_Jac_h_columns) <= n_z
    @assert length(sparsity_Jac_h_rows) == length(sparsity_Jac_h_columns)

    if build_Hessian
        @assert maximum(sparsity_Hes_L_rows) <= n_z
        @assert maximum(sparsity_Hes_L_columns) <= n_z
        @assert length(sparsity_Hes_L_rows) == length(sparsity_Hes_L_columns)
    end

    # Create the evaluator.
    return PMCMC_OCP_Evaluator(options,
        dimensions,
        indices,
        data,
        helpers,
        z_sets,
        h_bounds,
        sparsity_Jac_h,
        sparsity_Hes_L,
        nzrange_Jac_h,
        nzrange_Hes_L,
        backends,
        thread_cache,
        global_cache)
end