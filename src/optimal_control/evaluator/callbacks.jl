# Wrappers that allow to pack/unpack z by passing the evaluator.
pack_z(U::AbstractMatrix, X::AbstractArray, Y::AbstractArray, e::PMCMC_OCP_Evaluator; J_max::Union{Nothing,AbstractFloat}=nothing) = pack_z(U, X, Y, e.indices_U, e.indices_X, e.indices_Y, e.index_J_max, e.K, e.n_z, e.J_u; J_max=J_max)
unpack_z(z::AbstractVector, e::PMCMC_OCP_Evaluator) = unpack_z(z, e.indices_U, e.indices_X, e.indices_Y, e.index_J_max, e.K, e.n_u, e.n_x, e.n_y, e.H, e.J_u)

# This function returns the features that are available for the PMCMC_OCP_Evaluator; see MathOptInterface documentation.
function MOI.features_available(::PMCMC_OCP_Evaluator)
    return [:Grad, :Jac]
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
function MOI.eval_objective_gradient(e::PMCMC_OCP_Evaluator, grad::AbstractVector, z::AbstractVector)
    fill!(grad, 0.0)
    if e.J_u
        # Compute gradient of J(U) with respect to U.
        U_vec = view_U_vec(z, e.indices_U)
        g_nonzero = @views grad[e.indices_U]
        ForwardDiff.gradient!(g_nonzero, e.J_U_vec, U_vec)
    else
        # Set derivative of the epigraph variable J_max.
        grad[first(e.index_J_max)] = 1.0
    end
end

# The following function evaluates the constraint vector h(z).
function MOI.eval_constraint(e::PMCMC_OCP_Evaluator, h::AbstractVector, z::AbstractVector)
    Threads.@threads for k in 1:e.K
        # Get local variables
        if !e.J_u
            U, X_k, Y_k, J_max = unpack_z_k(z, k, e.indices_U, e.indices_X, e.indices_Y, e.index_J_max, e.n_u, e.n_x, e.n_y, e.H, e.J_u)
        else
            U, X_k, Y_k = unpack_z_k(z, k, e.indices_U, e.indices_X, e.indices_Y, e.index_J_max, e.n_u, e.n_x, e.n_y, e.H, e.J_u)
        end

        # Get local slices of the constraint vector h.
        h_dyn_x = @views h[e.indices_h_dynamics_x[k]]
        h_dyn_y = @views h[e.indices_h_dynamics_y[k]]
        h_scn = @views h[e.indices_h_scenario[k]]

        # Get data for this scenario.
        V_k = @views e.V[:, :, k]
        W_k = @views e.W[:, :, k]

        # Evaluate the dynamic constraints for the states and outputs.
        e.h_dynamics_x!(h_dyn_x, U, X_k, e.PMCMC_samples[k].theta, V_k)
        e.h_dynamics_y!(h_dyn_y, U, X_k, Y_k, e.PMCMC_samples[k].theta, W_k)

        # Evaluate the scenario constraints.
        e.h_scenario!(h_scn, U, X_k, Y_k)

        # Epigraph rows of constraint vector.
        if !e.J_u
            h_J_max = @views h[e.indices_h_J_max[k]]
            e.h_J_max!(h_J_max, U, X_k, Y_k, J_max)
        end
    end

    # Evaluate the input constraints h_u(U).
    U_vec = view_U_vec(z, e.indices_U)
    h_u = @views h[e.indices_h_u]
    e.h_u!(h_u, U_vec)
    return h
end

# The following function returns the sparsity pattern of the Jacobian of the constraints.
function MOI.jacobian_structure(e::PMCMC_OCP_Evaluator)
    return e.Jacobian_pattern_h
end

# The following function evaluates the Jacobian of the constraints.
function MOI.eval_constraint_jacobian(e::PMCMC_OCP_Evaluator, constraint_Jacobian_values::AbstractVector, z::AbstractVector)
    # Evaluate the dynamic, scenario and epigraph constraints for all scenarios thread-parallel and fill the values to the vector containing the non-zero entries of the global constraint Jacobian.
    Threads.@threads for k in 1:e.K
        # Get thread id and cache.
        thread_id = Threads.threadid()
        thread_cache = e.thread_cache[thread_id]

        # Get local decision vector.
        z_scenario = view_z_scenario(z, k, e.indices_U, e.indices_X, e.indices_Y, e.index_J_max, e.J_u)

        # Get data for this scenario.
        V_k = @views e.V[:, :, k]
        W_k = @views e.W[:, :, k]
        theta_k = @views e.PMCMC_samples[k].theta

        # Compute Jacobian of the dynamic constraints for the considered scenario.
        SparseDiffTools.forwarddiff_color_jacobian!(thread_cache.cache_h_dynamics_x.Jacobian, (h_dyn_x, z) -> e.h_dynamics_x!(h_dyn_x, z, theta_k, V_k), z_scenario, thread_cache.cache_h_dynamics_x.autodiff_cache)
        SparseDiffTools.forwarddiff_color_jacobian!(thread_cache.cache_h_dynamics_y.Jacobian, (h_dyn_y, z) -> e.h_dynamics_y!(h_dyn_y, z, theta_k, W_k), z_scenario, thread_cache.cache_h_dynamics_y.autodiff_cache)

        # Fill the values to the vector containing the non-zero entries of the constraint Jacobian.
        constraint_Jacobian_values[e.nzrange_h_dynamics_x[k]] .= thread_cache.cache_h_dynamics_x.Jacobian.nzval
        constraint_Jacobian_values[e.nzrange_h_dynamics_y[k]] .= thread_cache.cache_h_dynamics_y.Jacobian.nzval

        # Compute Jacobian of the scenario constraints for one scenario and fill the values to the vector containing the non-zero entries of the constraint Jacobian.
        SparseDiffTools.forwarddiff_color_jacobian!(thread_cache.cache_h_scenario.Jacobian, e.h_scenario!, z_scenario, thread_cache.cache_h_scenario.autodiff_cache)
        constraint_Jacobian_values[e.nzrange_h_scenario[k]] .= thread_cache.cache_h_scenario.Jacobian.nzval

        # Compute the Jacobian of the epigraph constraint J^[k] - J_max <= 0 (if used) and fill the values to the vector containing the non-zero entries of the constraint Jacobian.
        if !e.J_u
            SparseDiffTools.forwarddiff_color_jacobian!(thread_cache.cache_h_J_max.Jacobian, e.h_J_max!, z_scenario, thread_cache.cache_h_J_max.autodiff_cache)
            constraint_Jacobian_values[e.nzrange_h_J_max[k]] .= thread_cache.cache_h_J_max.Jacobian.nzval
        end
    end

    # Evaluate the Jacobian of the input constraints h_u(U) and fill the values to the vector containing the non-zero entries of the constraint Jacobian.
    U_vec = view_U_vec(z, e.indices_U)
    SparseDiffTools.forwarddiff_color_jacobian!(e.cache_h_u.Jacobian, e.h_u!, U_vec, e.cache_h_u.autodiff_cache)
    constraint_Jacobian_values[e.nzrange_h_u] .= e.cache_h_u.Jacobian.nzval

    return constraint_Jacobian_values
end