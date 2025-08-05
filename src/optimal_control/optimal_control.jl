"""
    solve_PMCMC_OCP(PMCMC_samples::Vector{PMCMC_sample}, f_theta::Function, g_theta::Function, sample_v_theta::Function, sample_w_theta::Function, H, J::Function, h_scenario::Function, h_u::Function; J_u=false, X_t=nothing, V=nothing, W=nothing, U_init=nothing, PMCMC_samples_pre_solve=nothing, K_warmup=0, solver_opts=nothing, print_progress=true)

Solve the optimal control problem of the following form:

``\\min_{u_{0:H},\\; \\overline{J_H}} \\overline{J_H}``

subject to: 
```math
\\begin{aligned}
\\forall k, &\\forall t \\\\
x_t^{[k]} &= f_{\\theta^{[k]}}(x_{t-1}^{[k]}, u_{t-1}) + v_{t-1}^{[k]}, \\\\
y_t^{[k]} &= g_{\\theta^{[k]}}(x_t^{[k]}, u_t) + w_t^{[k]}, \\\\
J_H^{[k]} &= J_H(u_{0:H}, x_{0:H}^{[k]}, y_{0:H}^{[k]}) \\leq \\overline{J_H}, \\\\
h_{\\mathrm{scenario}}(&u_{0:H},x_{0:H}^{[k]},y_{0:H}^{[k]}) \\leq 0. \\\\
h_{u}(&u_{0:H}) \\leq 0.
\\end{aligned}
```

# Arguments
- `PMCMC_samples`: PMCMC samples
- `f_theta`: state transition function parametrized by theta; has inputs (theta, x, u)
- `g_theta`: measurement function parametrized by theta; has inputs (theta, x, u)
- `sample_v_theta`: function that returns N samples from the process noise distribution parametrized by theta; has input (theta, N); only used if V is not passed
- `sample_w_theta`: function that returns N samples from the measurement noise distribution parametrized by theta; has input (theta, N); only used if W is not passed
- `H`: horizon of the OCP
- `J`: function with input arguments (``u_{1:H}``, ``x_{1:H}``, ``y_{1:H}``) (or ``u_{1:H}`` if `J_u` is set true) that returns the cost to be minimized
- `h_scenario`: function with input arguments (``u_{1:H}``, ``x_{1:H}``, ``y_{1:H}``) that returns the constraint vector belonging to a scenario; a feasible solution must satisfy ``h_{\\mathrm{scenario}} \\leq 0`` for all scenarios.
- `h_u`: function with input argument ``u_{1:H}`` that returns the constraint vector for the control inputs; a feasible solution satisfy ``h_u \\leq 0``.
- `J_u`: set to true if cost depends only on inputs ``u_{1:H}` - this accelerates the optimization
- `X_t`: vector with K * n_x elements containing the initial state of all model - if not provided, the initial states are sampled based on the particles in the PMCMC samples
- `V`: array of dimension n_x x H x K that contains the process noise for all models and all timesteps - if not provided, the noise is sampled using the function `sample_v_theta`
- `W`: array of dimension n_y x H x K that contains the measurement noise for all models and all timesteps - if not provided, the noise is sampled using the function `sample_w_theta`
- `U_init`: initial guess for the input trajectory
- `PMCMC_samples_pre_solve`: if provided, an initial guess for the input trajectory is obtained by solving an OCP with the samples in `PMCMC_pre_solve` only; they must be independent of `PMCMC_samples`
- `K_warmup`: if `K_warmup > 0` and `PMCMC_pre_solve` is provided, an initial guess for the the input trajectory is obtained in a two stage process: first, an OCP with only `K_warmup` samples from `PMCMC_samples_pre_solve` is solved and then an OCP with all samples in `PMCMC_samples_pre_solve`
- `solver_opts`: SolverOptions struct containing options of the solver
- `print_progress`: if set to true, the progress is printed

# Returns
- `U_opt`: optimal input trajectory
- `X_opt`: state trajectories for all scenarios, reshaped to a 3D array of dimension n_x x H x K
- `Y_opt`: optimal output trajectories for all scenarios, reshaped to a 3D array of dimension n_y x H x K
- `J_opt`: optimal cost
- `solve_successful`: true if the optimization was successful, false otherwise
- `iterations`: number of iterations of the solver
"""
function solve_PMCMC_OCP(PMCMC_samples::Vector{PMCMC_sample}, f_theta::Function, g_theta::Function, sample_v_theta::Function, sample_w_theta::Function, H, J::Function, h_scenario::Function, h_u::Function; J_u=false, X_t=nothing, V=nothing, W=nothing, U_init=nothing, PMCMC_samples_pre_solve=nothing, K_warmup=0, solver_opts=nothing, print_progress=true)
    # Time optimization.
    optimization_timer = time()

    # Get number of states, etc.
    K = size(PMCMC_samples, 1)
    n_u = size(PMCMC_samples[1].u_m1, 1)
    n_x = size(PMCMC_samples[1].x_m1, 1)
    n_y = size(sample_w_theta(PMCMC_samples[1].theta, 1), 1)

    # Sample initial states if not provided.
    if X_t === nothing
        X_t = Array{Float64}(undef, n_x, K)
        for k in 1:K
            # Sample state and propagate.
            star = sample(1:length(PMCMC_samples[k].w_m1), Weights(PMCMC_samples[k].w_m1))
            x_m1 = PMCMC_samples[k].x_m1[:, star]
            X_t[:, k] .= f_theta(PMCMC_samples[k].theta, x_m1, PMCMC_samples[k].u_m1) .+ sample_v_theta(PMCMC_samples[k].theta, 1)
        end
    end

    # Sample process noise array V if not provided.
    if V === nothing
        V = Array{Float64}(undef, n_x, H, K)
        for k in 1:K
            V[:, :, k] = sample_v_theta(PMCMC_samples[k].theta, H)
        end
    end

    # Sample measurement noise array W if not provided.
    if W === nothing
        W = Array{Float64}(undef, n_y, H, K)
        for k in 1:K
            W[:, :, k] = sample_w_theta(PMCMC_samples[k].theta, H)
        end
    end

    # Determine initialization.
    # With a good initialization the runtime of the optimization can be reduced significantly.
    # If an additional set of samples `PMCMC_samples_pre_solve` is provided (must be independent of `PMCMC_samples`), the OCP is solved with the samples in `PMCMC_samples_pre_solve` first to obtain an initialization for the problem the samples in `PMCMC_samples`.
    # If additionally `K_warmup` is provided, a problem that only considers `K_warmup` randomly selected scenarios of `PMCMC_samples_pre_solve` is solved first to obtain an initialization for the problem with all scenarios in `PMCMC_samples_pre_solve`.
    if !(PMCMC_samples_pre_solve === nothing)
        if K_warmup > 0
            # Sample the K_warmup scenarios that are considered for the initialization.
            warmup_samples = sample(1:size(PMCMC_samples_pre_solve, 1), K_warmup)

            if print_progress
                println("###### Started pre-solving step")
            end

            # Solve OCP with K_warmup samples from PMCMC_samples_pre_solve.
            U_init = solve_PMCMC_OCP(PMCMC_samples_pre_solve[warmup_samples], f_theta, g_theta, sample_v_theta, sample_w_theta, H, J, h_scenario, h_u; J_u=J_u, U_init=U_init, solver_opts=solver_opts, print_progress=print_progress)[1]

            # Solve OCP with all samples from PMCMC_samples_pre_solve.
            U_init = solve_PMCMC_OCP(PMCMC_samples_pre_solve, f_theta, g_theta, sample_v_theta, sample_w_theta, H, J, h_scenario, h_u; J_u=J_u, U_init=U_init, solver_opts=solver_opts, print_progress=print_progress)[1]

            if print_progress
                println("###### Pre-solving step complete, switching back to the original problem")
            end
        else
            if print_progress
                println("###### Started pre-solving step")
            end

            # Solve OCP with all samples from PMCMC_samples_pre_solve.
            U_init = solve_PMCMC_OCP(PMCMC_samples_pre_solve, f_theta, g_theta, sample_v_theta, sample_w_theta, H, J, h_scenario, h_u; J_u=J_u, U_init=U_init, solver_opts=solver_opts, print_progress=print_progress)[1]

            if print_progress
                println("###### Pre-solving step complete, switching back to the original problem")
            end
        end
    elseif U_init === nothing
        U_init = zeros(n_u, H)
    end

    # Determine initial guess for X and Y.
    X_init = Array{Float64}(undef, n_x, H, K) # initial guess for X
    Y_init = Array{Float64}(undef, n_y, H, K) # initial guess for Y
    for k in 1:K
        # Get current model.
        f(x, u) = f_theta(PMCMC_samples[k].theta, x, u)
        g(x, u) = g_theta(PMCMC_samples[k].theta, x, u)

        X_init[:, 1, k] .= X_t[:, k]
        for t in 2:H
            X_init[:, t, k] = f(X_init[:, t-1, k], U_init[:, t-1]) + V[:, t-1, k]
        end
        for t in 1:H
            Y_init[:, t, k] = g(X_init[:, t, k], U_init[:, t]) + W[:, t, k]
        end
    end

    # If the limited-memory Hessian approximation is used, the Hessian sparsity structure is not computed.
    build_Hessian = get(solver_opts, "hessian_approximation", "") != "limited-memory"

    # Create evaluator.
    evaluator = PMCMC_OCP_Evaluator(PMCMC_samples, V, W, X_t, f_theta, g_theta, J, J_u, h_scenario, h_u, n_u, n_x, n_y, H; build_Hessian=build_Hessian, deduplicate_Hessian=true)

    # Create model.
    model = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())

    # Add decision variables and bounds.
    z_indices = Vector{MOI.VariableIndex}(undef, evaluator.dimensions.n_z)

    for (i, set) in enumerate(evaluator.z_sets)
        z_indices[i] = MOI.add_variable(model)
        MOI.add_constraint(model, z_indices[i], set)
    end

    # Add constraints.
    MOI.set(model, MOI.NLPBlock(), MOI.NLPBlockData(evaluator.h_bounds, evaluator, true))

    # Goal: Minimize the objective.
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # Set initial state for the decision vector.
    if J_u
        z_init = pack_z(U_init, X_init, Y_init, evaluator)
    else
        z_init = pack_z(U_init, X_init, Y_init, evaluator; J_max=0.0)
    end

    for (index, value) in zip(z_indices, z_init)
        MOI.set(model, MOI.VariablePrimalStart(), index, value)
    end

    # Create solver.
    solver = Ipopt.Optimizer()

    # Set options.
    if !(solver_opts === nothing)
        for (opt, val) in solver_opts
            MOI.set(solver, MOI.RawOptimizerAttribute(opt), val)
        end
    end

    if !print_progress
        MOI.set(solver, MOI.Silent(), true)
    end

    # Solve OCP.
    if print_progress
        println("### Started optimization algorithm")
    end

    MOI.optimize!(solver, model)
    time_optimization = time() - optimization_timer

    if print_progress
        @printf("### Optimization complete\nRuntime: %.2f s\n", time_optimization)
    end

    # status = MOI.get(model, MOI.TerminationStatus())
    z_opt = MOI.get(solver, MOI.VariablePrimal(), z_indices)

    # Extract the solution.
    U_opt, X_opt, Y_opt = unpack_z(z_opt, evaluator)[1:3]
    J_opt = MOI.get(solver, MOI.ObjectiveValue())
    termination_status = MOI.get(solver, MOI.TerminationStatus())
    primal_status = MOI.get(solver, MOI.PrimalStatus())
    solve_successful = (termination_status == MOI.OPTIMAL && primal_status == MOI.FEASIBLE_POINT)
    iterations = MOI.get(solver, MOI.BarrierIterations())

    if !solve_successful
        @warn ("The optimization did not converge to an optimal and feasible solution. " *
               "Termination status: $termination_status, Primal status: $primal_status.")
    end

    return U_opt, X_opt, Y_opt, J_opt, solve_successful, iterations
end

"""
    solve_PG_OCP_greedy_guarantees(PMCMC_samples::Vector{PMCMC_sample}, n_x, f_theta::Function, g_theta::Function, sample_v_theta::Function, sample_w_theta::Function, H, J::Function, h_scenario::Function, h_u::Function, β; J_u=false, X_t=nothing, V=nothing, W=nothing, U_init=nothing, PMCMC_samples_pre_solve=nothing, K_warmup=0, delta_tol=1e-6, solver_opts=nothing, print_progress=true)

Solve the sample-based optimal control problem using Ipopt and determine a support sub-sample with cardinality s via a greedy constraint removal.
Based on the cardinality s, a bound on the probability that the incurred cost exceeds the worst-case cost or that the constraints are violated when the input trajectory u_{0:H} is applied to the unknown system is calculated.

``\\min_{u_{0:H},\\; \\overline{J_H}} \\overline{J_H}``

subject to: 
```math
\\begin{aligned}
\\forall k, &\\forall t \\\\
x_t^{[k]} &= f_{\\theta^{[k]}}(x_{t-1}^{[k]}, u_{t-1}) + v_{t-1}^{[k]}, \\\\
y_t^{[k]} &= g_{\\theta^{[k]}}(x_t^{[k]}, u_t) + w_t^{[k]}, \\\\
J_H^{[k]} &= J_H(u_{0:H}, x_{0:H}^{[k]}, y_{0:H}^{[k]}) \\leq \\overline{J_H}, \\\\
h_{\\mathrm{scenario}}(&u_{0:H},x_{0:H}^{[k]},y_{0:H}^{[k]}) \\leq 0. \\\\
h_{u}(&u_{0:H}) \\leq 0.
\\end{aligned}
```

# Arguments
- `PMCMC_samples`: PMCMC samples
- `n_x`: number of states
- `f_theta`: state transition function parametrized by theta; has inputs (theta, x, u)
- `g_theta`: measurement function parametrized by theta; has inputs (theta, x, u)
- `sample_v_theta`: function that returns N samples from the process noise distribution parametrized by theta; has input (theta, N); only used if V is not passed
- `sample_w_theta`: function that returns N samples from the measurement noise distribution parametrized by theta; has input (theta, N); only used if W is not passed
- `H`: horizon of the OCP
- `J`: function with input arguments (``u_{1:H}``, ``x_{1:H}``, ``y_{1:H}``) (or ``u_{1:H}`` if `J_u` is set true) that returns the cost to be minimized
- `h_scenario`: function with input arguments (``u_{1:H}``, ``x_{1:H}``, ``y_{1:H}``) that returns the constraint vector belonging to a scenario; a feasible solution must satisfy ``h_{\\mathrm{scenario}} \\leq 0`` for all scenarios.
- `h_u`: function with input argument ``u_{1:H}`` that returns the constraint vector for the control inputs; a feasible solution satisfy ``h_u \\leq 0``.
- `β`: confidence parameter
- `J_u`: set to true if cost depends only on inputs ``u_{1:H}` - this accelerates the optimization
- `X_t`: vector with K * n_x elements containing the initial state of all models - if not provided, the initial states are sampled based on the PGS samples
- `V`: array of dimension n_x x H x K that contains the process noise for all models and all timesteps - if not provided, the noise is sampled based on the PGS samples
- `W`: array of dimension n_y x H x K that contains the measurement noise for all models and all timesteps - if not provided, the noise is sampled based on the provided `R`
- `U_init`: initial guess for the input trajectory
- `PMCMC_samples_pre_solve`: if provided, an initial guess for the input trajectory is obtained by solving an OCP with the samples in `PMCMC_pre_solve` only; they must be independent of `PMCMC_samples`
- `K_warmup`: if `K_warmup > 0` and `PMCMC_pre_solve` is provided, an initial guess for the the input trajectory is obtained in a two stage process: first, an OCP with only `K_warmup` samples from `PMCMC_samples_pre_solve` is solved and then an OCP with all samples in `PMCMC_samples_pre_solve`
- `delta_tol`: maximum absolute distance below which two solutions (input trajectories and maximum cost) are considered identical
- `solver_opts`: SolverOptions struct containing options of the solver
- `print_progress`: if set to true, the progress is printed

# Returns
- `U_opt`: optimal input trajectory
- `X_opt`: state trajectories for all scenarios, reshaped to a 3D array of dimension n_x x H x K
- `Y_opt`: optimal output trajectories for all scenarios, reshaped to a 3D array of dimension n_y x H x K
- `J_opt`: optimal cost
- `s`: cardinality of the found support sub-sample
- `epsilon_prob`: bound on the probability that the incurred cost exceeds the worst-case cost or that the constraints are violated when the input trajectory u_{0:H} is applied to the unknown system
- `epsilon_perc`: bound on the probability that the incurred cost exceeds the worst-case cost or that the constraints are violated when the input trajectory u_{0:H} is applied to the unknown system in percent
- `time_first_solve`: time it took to solve the OCP for the first time
- `time_guarantees`: time it took to compute the guarantees
- `num_failed_optimizations`: number of failed optimizations during the computation of the guarantees
"""
function solve_PG_OCP_greedy_guarantees(PMCMC_samples::Vector{PMCMC_sample}, n_x, f_theta::Function, g_theta::Function, sample_v_theta::Function, sample_w_theta::Function, H, J::Function, h_scenario::Function, h_u::Function, β::AbstractFloat; J_u=false, X_t=nothing, V=nothing, W=nothing, U_init=nothing, PMCMC_samples_pre_solve=nothing, K_warmup=0, delta_tol=1e-6, solver_opts=nothing, print_progress=true)
    # Time first optimization.
    first_solve_timer = time()

    # Get number of states, etc.
    K = size(PMCMC_samples, 1)
    n_u = size(PMCMC_samples[1].u_m1, 1)
    n_x = size(PMCMC_samples[1].x_m1, 1)
    n_y = size(sample_w_theta(PMCMC_samples[1].theta, 1), 1)

    # Sample initial states if not provided.
    if X_t === nothing
        X_t = Array{Float64}(undef, n_x * K)
        for k in 1:K
            # Sample state and propagate.
            star = sample(1:length(PMCMC_samples[k].w_m1), Weights(PMCMC_samples[k].w_m1))
            x_m1 = PMCMC_samples[k].x_m1[:, star]
            X_t[(k-1)*n_x+1:n_x*k] .= f_theta(PMCMC_samples[k].theta, x_m1, PMCMC_samples[k].u_m1) .+ sample_v_theta(PMCMC_samples[k].theta, 1)
        end
    end

    # Sample process noise array V if not provided.
    if V === nothing
        V = Array{Float64}(undef, n_x, H, K)
        for k in 1:K
            V[:, :, k] = sample_v_theta(PMCMC_samples[k].theta, H)
        end
    end

    # Sample measurement noise array W if not provided.
    if W === nothing
        W = Array{Float64}(undef, n_y, H, K)
        for k in 1:K
            W[:, :, k] = sample_w_theta(PMCMC_samples[k].theta, H)
        end
    end

    # Determine initialization.
    # With a good initialization the runtime of the optimization can be reduced significantly.
    # If an additional set of samples `PMCMC_samples_pre_solve` is provided (must be independent of `PMCMC_samples`), the OCP is solved with the samples in `PMCMC_samples_pre_solve` first to obtain an initialization for the problem the samples in `PMCMC_samples`.
    # If additionally `K_warmup` is provided, a problem that only considers `K_warmup` randomly selected scenarios of `PMCMC_samples_pre_solve` is solved first to obtain an initialization for the problem with all scenarios in `PMCMC_samples_pre_solve`.
    # Determine initialization.
    # With a good initialization the runtime of the optimization can be reduced significantly.
    # If an additional set of samples `PMCMC_samples_pre_solve` is provided (must be independent of `PMCMC_samples`), the OCP is solved with the samples in `PMCMC_samples_pre_solve` first to obtain an initialization for the problem the samples in `PMCMC_samples`.
    # If additionally `K_warmup` is provided, a problem that only considers `K_warmup` randomly selected scenarios of `PMCMC_samples_pre_solve` is solved first to obtain an initialization for the problem with all scenarios in `PMCMC_samples_pre_solve`.
    if !(PMCMC_samples_pre_solve === nothing)
        if K_warmup > 0
            # Sample the K_warmup scenarios that are considered for the initialization.
            warmup_samples = sample(1:size(PMCMC_samples_pre_solve, 1), K_warmup)

            if print_progress
                println("###### Started pre-solving step")
            end

            # Solve OCP with K_warmup samples from PMCMC_samples_pre_solve.
            U_init = solve_PMCMC_OCP(PMCMC_samples_pre_solve[warmup_samples], f_theta, g_theta, sample_v_theta, sample_w_theta, H, J, h_scenario, h_u; J_u=J_u, U_init=U_init, K_warmup=0, solver_opts=solver_opts, print_progress=print_progress)[1]

            # Solve OCP with all samples from PMCMC_samples_pre_solve.
            U_init = solve_PMCMC_OCP(PMCMC_samples_pre_solve, f_theta, g_theta, sample_v_theta, sample_w_theta, H, J, h_scenario, h_u; J_u=J_u, U_init=U_init, K_warmup=0, solver_opts=solver_opts, print_progress=print_progress)[1]

            if print_progress
                println("###### Pre-solving step complete, switching back to the original problem")
            end
        else
            if print_progress
                println("###### Started pre-solving step")
            end

            # Solve OCP with all samples from PMCMC_samples_pre_solve.
            U_init = solve_PMCMC_OCP(PMCMC_samples_pre_solve, f_theta, g_theta, sample_v_theta, sample_w_theta, H, J, h_scenario, h_u; J_u=J_u, U_init=U_init, K_warmup=0, solver_opts=solver_opts, print_progress=print_progress)[1]

            if print_progress
                println("###### Pre-solving step complete, switching back to the original problem")
            end
        end
    elseif U_init === nothing
        U_init = zeros(n_u, H)
    end

    num_failed_optimizations = 0 # number of failed optimizations during the computation of the guarantees

    # Find optimal trajectory.
    if print_progress
        println("### Started optimization of fully constrained problem")
    end

    # Solve the OCP.
    U_opt, X_opt, Y_opt, J_opt, solve_successful, iterations = solve_PMCMC_OCP(PMCMC_samples, f_theta, g_theta, sample_v_theta, sample_w_theta, H, J, h_scenario, h_u; J_u=J_u, X_t=X_t, V=V, W=W, U_init=U_init, solver_opts=solver_opts, print_progress=print_progress)

    # Determine guarantees.
    # If a feasible U_opt is found, probabilistic constraint satisfaction guarantees are derived by greedily removing constraints to determine a support sub-sample S.
    if solve_successful
        if print_progress
            println("### Optimization of fully constrained problem successful, continouing with computation of guarantees")
        end

        time_first_solve = time() - first_solve_timer

        # Time computation of guarantees.
        guarantees_timer = time()

        # Determine support sub-samples and guarantees for the generalization of the resulting input trajectory.
        println("### Started search for support sub-sample")

        # Reduce number of iterations - if the number of iterations of the original OCP is exceeded, the solution will likely be different, and the optimization can be stopped.
        solver_opts["max_iter"] = 2 * iterations

        # Sort scenarios according to the distance to the constraint boundary - removing the scenarios with the largest distance to the constraint boundary first usually yields better results.
        h_scenario_max = Array{Float64}(undef, K) # minimum distance of the scenarios to the constraint boundary
        for i in 1:K
            h_scenario_max[i] = maximum(h_scenario(U_opt, X_opt[:, :, i], Y_opt[:, :, i]))
        end
        scenarios_sorted = sortperm(h_scenario_max; rev=true) # sort scenarios

        # Pre-allocate.
        U_opt_temp = Array{Float64}(undef, n_u, H - 1)
        active_scenarios = scenarios_sorted

        # Greedily remove constraints and check whether the solution changes to determine a support sub-sample.
        for i in 1:K
            # Print progress.
            @printf("Started optimization with new constraint set\nIteration: %i/%i\n", i, K)

            # Temporaily remove the constraints corresponding to the PG samples with index i from the constraint set.
            temp_scenarios = active_scenarios[active_scenarios.!==scenarios_sorted[i]]

            # Get the initial states of the active scenarios.
            X_t_temp = Array{Float64}(undef, n_x * length(temp_scenarios))
            for k in eachindex(temp_scenarios)
                X_t_temp[(k-1)*n_x+1:n_x*k] = X_t[(temp_scenarios[k]-1)*n_x+1:n_x*temp_scenarios[k]]
            end

            # Solve the OCP with reduced constraint set.
            U_opt_temp, J_opt_temp, solve_successful_temp = solve_PMCMC_OCP(PMCMC_samples[temp_scenarios], f_theta, g_theta, sample_v_theta, sample_w_theta, H, J, h_scenario, h_u; J_u=J_u, X_t=X_t, V=V[:, :, temp_scenarios], W=W[:, :, temp_scenarios], U_init=U_init, solver_opts=solver_opts, print_progress=print_progress)

            # If the optimization is successful and the solution does not change, permanently remove the constraints corresponding to the PG samples with index i from the constraint set.
            # A valid subsample has the same local minimum. However, since the numerical solver does not reach this minimum exactly, a threshold value is used here to check whether the solutions are the same.
            if solve_successful_temp && all(abs.(U_opt_temp - U_opt) .< delta_tol) && all(abs.(J_opt_temp - J_opt) .< delta_tol)
                active_scenarios = temp_scenarios
            elseif !solve_successful_temp
                @warn "Optimization with temporarily removed constraints failed. Proceeding with next candidate for a support sub-sample."
                num_failed_optimizations += 1
            end
        end

        # Determine the cardinality of the support sub-sample.
        s = length(active_scenarios)

        # Based on the cardinality of the support sub-sample, determine the parameter ϵ. 
        # 1-ϵ corresponds to a bound on the probability that the incurred cost exceeds the worst-case cost or that the constraints are violated when the input trajectory u_{0:H} is applied to the unknown system.
        epsilon_prob = epsilon(s, K, β)
        epsilon_perc = epsilon_prob * 100

        # Print s, ϵ, and runtime.
        time_guarantees = time() - guarantees_timer
        @printf("### Support sub sample found\nCardinality of the support sub-sample (s): %i\nMax. constraint violation probability (1-epsilon): %.2f %%\nTime to compute u*: %.2f s\nTime to compute 1-epsilon: %.2f s\n", s, 100 - epsilon_perc, time_first_solve, time_guarantees)
    else
        # In case the initial problem is infeasible, skip the computation of guarantees.
        @warn "No feasible solution found for the initial problem. Skipping computation of guarantees."
        time_first_solve = NaN
        J_opt = NaN
        s = NaN
        epsilon_prob = NaN
        epsilon_perc = NaN
        time_guarantees = NaN
    end
    return U_opt, X_opt, Y_opt, J_opt, s, epsilon_prob, epsilon_perc, time_first_solve, time_guarantees, num_failed_optimizations
end