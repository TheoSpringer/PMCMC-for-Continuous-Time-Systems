# Wrappers that allow to pack/unpack z by passing the evaluator.
pack_z(U::AbstractMatrix, X::AbstractArray, Y::AbstractArray, e::PMCMC_OCP_Evaluator; J_max::Union{Nothing,AbstractFloat}=nothing) = pack_z(U, X, Y, e.indices, e.dimensions; J_max=J_max)
unpack_z(z::AbstractVector, e::PMCMC_OCP_Evaluator) = unpack_z(z, e.indices, e.dimensions)

# This function returns the features that are available for the PMCMC_OCP_Evaluator; see MathOptInterface documentation.
function MOI.features_available(e::PMCMC_OCP_Evaluator)
    if e.Hessian_build
        return [:Grad, :Jac, :Hess]
    else
        return [:Grad, :Jac]
    end
end

# The following function is required by MathOptInterface.
# It is called once the evaluator is added to the model.
# This function only checks if the requested features are available.
function MOI.initialize(e::PMCMC_OCP_Evaluator, requested::Vector{Symbol})
    available = Set(MOI.features_available(e))
    bad = setdiff(requested, available)

    if !isempty(bad)
        throw(MOI.UnsupportedFeature(
            "Requested feature(s) $(collect(bad)) not supported." * "Available: $(collect(avail))"))
    end
    return nothing
end

# The following function evaluates the objective function.
function MOI.eval_objective(e::PMCMC_OCP_Evaluator, z::AbstractVector)
    if e.dimensions.J_u
        # J(U) is used as the objective function.
        U_vec = view_U_vec(z, e.indices.U)
        return e.helpers.eval_J_u(U_vec)
    else
        # Epigraph notation is used.
        return z[first(e.indices.J_max)]
    end
end

# The following function evaluates the gradient of the objective function.
function MOI.eval_objective_gradient(e::PMCMC_OCP_Evaluator, grad::AbstractVector, z::AbstractVector)
    fill!(grad, 0.0)
    if e.dimensions.J_u
        # Compute gradient of J(U) with respect to U.
        U_vec = view_U_vec(z, e.indices.U)
        g_nonzero = @views grad[e.indices.U]
        ForwardDiff.gradient!(g_nonzero, e.helpers.eval_J_u, U_vec)
    else
        # Set derivative of the epigraph variable J_max.
        grad[first(e.indices.J_max)] = 1.0
    end
end

# The following function evaluates the constraint vector h(z).
function MOI.eval_constraint(e::PMCMC_OCP_Evaluator, h::AbstractVector, z::AbstractVector)
    Threads.@threads for k in 1:e.dimensions.K
        # Get local variables
        if !e.dimensions.J_u
            U, X_k, Y_k, J_max = unpack_z_k(z, k, e.indices, e.dimensions)
        else
            U, X_k, Y_k = unpack_z_k(z, k, e.indices, e.dimensions)
        end

        # Get local slices of the constraint vector h.
        h_dyn_x = @views h[e.indices.h_dynamics_x[k]]
        h_dyn_y = @views h[e.indices.h_dynamics_y[k]]
        h_scn = @views h[e.indices.h_scenario[k]]

        # Get data for this scenario.
        V_k = @views e.data.V[:, :, k]
        W_k = @views e.data.W[:, :, k]

        # Evaluate the dynamic constraints for the states and outputs.
        e.helpers.h_dynamics_x!(h_dyn_x, U, X_k, e.data.PMCMC_samples[k].theta, V_k)
        e.helpers.h_dynamics_y!(h_dyn_y, U, X_k, Y_k, e.data.PMCMC_samples[k].theta, W_k)

        # Evaluate the scenario constraints.
        e.helpers.h_scenario!(h_scn, U, X_k, Y_k)

        # Epigraph rows of constraint vector.
        if !e.dimensions.J_u
            h_J_max = @views h[e.indices.h_J_max[k]]
            e.helpers.h_J_max!(h_J_max, U, X_k, Y_k, J_max)
        end
    end

    # Evaluate the input constraints h_u(U).
    U_vec = view_U_vec(z, e.indices)
    h_u = @views h[e.indices.h_u]
    e.helpers.h_u!(h_u, U_vec)
    return h
end

# The following function returns the sparsity pattern of the Jacobian of the constraints.
function MOI.jacobian_structure(e::PMCMC_OCP_Evaluator)
    return e.Jacobian_pattern_h
end

# The following function evaluates the Jacobian of the constraints.
function MOI.eval_constraint_jacobian(e::PMCMC_OCP_Evaluator, constraint_Jacobian_values::AbstractVector, z::AbstractVector)
    # Evaluate the dynamic, scenario and epigraph constraints for all scenarios thread-parallel and fill the values to the vector containing the non-zero entries of the global constraint Jacobian.
    Threads.@threads for k in 1:e.dimensions.K
        # Get thread id and cache.
        thread_id = Threads.threadid()
        thread_cache = e.thread_cache[thread_id]

        # Get local decision vector.
        z_scenario = view_z_scenario(z, k, e.indices, e.dimensions)

        # Get data for this scenario.
        V_k = @views e.data.V[:, :, k]
        W_k = @views e.data.W[:, :, k]
        theta_k = @views e.data.PMCMC_samples[k].theta

        # Compute Jacobian of the dynamic constraints for the considered scenario.
        SparseDiffTools.forwarddiff_color_jacobian!(thread_cache.Jacobian_h_dynamics_x.buffer, (h_dyn_x, z) -> e.helpers.h_dynamics_x!(h_dyn_x, z, theta_k, V_k), z_scenario, thread_cache.Jacobian_h_dynamics_x.autodiff_cache)
        SparseDiffTools.forwarddiff_color_jacobian!(thread_cache.Jacobian_h_dynamics_y.buffer, (h_dyn_y, z) -> e.helpers.h_dynamics_y!(h_dyn_y, z, theta_k, W_k), z_scenario, thread_cache.Jacobian_h_dynamics_y.autodiff_cache)

        # Fill the values to the vector containing the non-zero entries of the constraint Jacobian.
        constraint_Jacobian_values[e.nzrange_Jacobian_h.h_dynamics_x[k]] .= thread_cache.Jacobian_h_dynamics_x.buffer.nzval
        constraint_Jacobian_values[e.nzrange_Jacobian_h.h_dynamics_y[k]] .= thread_cache.Jacobian_h_dynamics_y.buffer.nzval

        # Compute Jacobian of the scenario constraints for the considered scenario and fill the values to the vector containing the non-zero entries of the constraint Jacobian.
        SparseDiffTools.forwarddiff_color_jacobian!(thread_cache.Jacobian_h_scenario.buffer, e.helpers.h_scenario!, z_scenario, thread_cache.Jacobian_h_scenario.autodiff_cache)
        constraint_Jacobian_values[e.nzrange_Jacobian_h.h_scenario[k]] .= thread_cache.Jacobian_h_scenario.buffer.nzval

        # Compute the Jacobian of the epigraph constraint J^[k] - J_max <= 0 (if used) and fill the values to the vector containing the non-zero entries of the constraint Jacobian.
        if !e.dimensions.J_u
            SparseDiffTools.forwarddiff_color_jacobian!(thread_cache.Jacobian_h_J_max.buffer, e.helpers.h_J_max!, z_scenario, thread_cache.Jacobian_h_J_max.autodiff_cache)
            constraint_Jacobian_values[e.nzrange_Jacobian_h.h_J_max[k]] .= thread_cache.Jacobian_h_J_max.buffer.nzval
        end
    end

    # Evaluate the Jacobian of the input constraints h_u(U) and fill the values to the vector containing the non-zero entries of the constraint Jacobian.
    U_vec = view_U_vec(z, e.indices)
    SparseDiffTools.forwarddiff_color_jacobian!(e.cache_Jacobian_h_u.buffer, e.helpers.h_u!, U_vec, e.cache_Jacobian_h_u.autodiff_cache)
    constraint_Jacobian_values[e.nzrange_Jacobian_h.h_u] .= e.cache_Jacobian_h_u.buffer.nzval

    return constraint_Jacobian_values
end

# The following function returns the sparsity pattern of the Hessian of the Lagrangian.
function MOI.hessian_lagrangian_structure(e::PMCMC_OCP_Evaluator)
    return e.Hessian_pattern_L
end

# The following function evaluates the Hessian of the Lagrangian.
function MOI.eval_hessian_lagrangian(e::PMCMC_OCP_Evaluator, hessian_Lagrangian_values::AbstractVector, z::AbstractVector, lambda_objective, lambda_constraints::AbstractVector)
    # Evaluate the Hessian of the local Lagrangians for all scenarios thread-parallel and fill the values to the vector containing the non-zero entries of the global Hessian of the Lagrangian.
    Threads.@threads for k in 1:e.dimensions.K
        # Get thread id and cache.
        thread_id = Threads.threadid()
        thread_cache = e.thread_cache[thread_id]

        # Get local decision vector.
        z_scenario = view_z_scenario(z, k, e.indices, e.dimensions)

        # Get data for this scenario.
        V_k = @views e.data.V[:, :, k]
        W_k = @views e.data.W[:, :, k]
        theta_k = @views e.data.PMCMC_samples[k].theta

        # Extract the local Lagrange multipliers.
        lambda_h_dynamics_x = @views lambda_constraints[e.indices.h_dynamics_x[k]]
        lambda_h_dynamics_y = @views lambda_constraints[e.indices.h_dynamics_y[k]]
        lambda_h_scenario = @views lambda_constraints[e.indices.h_scenario[k]]

        # Compute Hessian of the local Lagrangian of the dynamic constraints for the considered scenario.
        SparseDiffTools.numauto_color_hessian!(thread_cache.Hessian_L_dynamics_x.buffer, (z) -> e.helpers.lagrangian_dynamics_x(lambda_h_dynamics_x, z, theta_k, V_k), z_scenario, thread_cache.Hessian_L_dynamics_x.autodiff_cache)
        SparseDiffTools.numauto_color_hessian!(thread_cache.Hessian_L_dynamics_y.buffer, (z) -> e.helpers.lagrangian_dynamics_y(lambda_h_dynamics_y, z, theta_k, W_k), z_scenario, thread_cache.Hessian_L_dynamics_y.autodiff_cache)

        # Fill the values to the vector containing the non-zero entries of the Hessian of the Lagrangian.
        hessian_Lagrangian_values[e.nzrange_Hessian_L.L_dynamics_x[k]] .= thread_cache.Hessian_L_dynamics_x.buffer.nzval
        hessian_Lagrangian_values[e.nzrange_Hessian_L.L_dynamics_y[k]] .= thread_cache.Hessian_L_dynamics_y.buffer.nzval

        # Compute Hessian of the local Lagrangian of the scenario constraints for the considered scenario and fill the values to the vector containing the non-zero entries of the Hessian of the Lagrangian.
        SparseDiffTools.numauto_color_hessian!(thread_cache.Hessian_L_scenario.buffer, (z) -> e.helpers.lagrangian_scenario(lambda_h_scenario, z), z_scenario, thread_cache.Hessian_L_scenario.autodiff_cache)
        hessian_Lagrangian_values[e.nzrange_Hessian_L.L_scenario[k]] .= thread_cache.Hessian_L_scenario.buffer.nzval

        # Compute the Hessian of the local Lagrangian of the epigraph constraint J^[k] - J_max <= 0 (if used) for the considered scenario and fill the values to the vector containing the non-zero entries of the Hessian of the Lagrangian.
        if !e.dimensions.J_u
            lambda_h_J_max = @views lambda_constraints[e.indices.h_J_max[k]]
            SparseDiffTools.numauto_color_hessian!(thread_cache.Hessian_L_J_max.buffer, (z) -> e.helpers.lagrangian_J_max(lambda_h_J_max, z), z_scenario, thread_cache.Hessian_L_J_max.autodiff_cache)
            hessian_Lagrangian_values[e.nzrange_Hessian_L.L_J_max[k]] .= thread_cache.Hessian_L_J_max.buffer.nzval
        end
    end
    # Evaluate the Hessian of the local Lagrangian of the input constraints h_u(U) and fill the values to the vector containing the non-zero entries of the Hessian of the Lagrangian.
    U_vec = view_U_vec(z, e.indices)
    lambda_h_u = @views lambda_constraints[e.indices.h_u]
    SparseDiffTools.numauto_color_hessian!(e.cache_Hessian_L_u.buffer, (U_vec) -> e.helpers.lagrangian_u(lambda_h_u, U_vec), U_vec, e.cache_Hessian_L_u.autodiff_cache)
    hessian_Lagrangian_values[e.nzrange_Hessian_L.L_u] .= e.cache_Hessian_L_u.buffer.nzval

    # Evaluate the Hessian of the objective function. If the epigraph notation is used, the Hessian of the objective function is zero and thus ignored.
    if e.dimensions.J_u
        SparseDiffTools.numauto_color_hessian!(e.cache_Hessian_J_u.buffer, (U_vec) -> e.helpers.eval_J_u(U_vec), U_vec, e.cache_Hessian_J_u.autodiff_cache)
        hessian_Lagrangian_values[e.nzrange_Hessian_L.L_J_u] .= lambda_objective .* e.cache_Hessian_J_u.buffer.nzval
    end
end
