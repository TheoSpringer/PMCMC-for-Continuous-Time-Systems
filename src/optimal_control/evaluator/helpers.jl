# This file contains definitions and helper functions that are called in the constructor of the PMCMC_OCP_Evaluator.

# The following struct contains the cached data for the automatic differentiation.
struct ADCache{CacheType,MatrixType}
    autodiff_cache::CacheType
    buffer::MatrixType
end

# The following struct contains the cached data for the automatic differentiation of the dynamic, scenario, and epigraph constraints.
struct ThreadCache
    cache_Jacobian_h_dynamics_x::ADCache
    cache_Jacobian_h_dynamics_y::ADCache
    cache_Jacobian_h_scenario::ADCache
    cache_Jacobian_h_J_max::Union{ADCache,Nothing}

    cache_Hessian_L_dynamics_x::Union{ADCache,Nothing}
    cache_Hessian_L_dynamics_y::Union{ADCache,Nothing}
    cache_Hessian_L_scenario::Union{ADCache,Nothing}
    cache_Hessian_L_J_max::Union{ADCache,Nothing}
end

# The following function builds the helper functions that evaluate the constraints 
function build_helpers(f_theta, g_theta, h_scenario, h_u, J, J_u, n_u::Int, n_x::Int, n_y::Int, H::Int; build_hessian::Bool=true)
    # Helper function that evaluates the dynamics constraints for the states over the whole horizon H for one scenario.
    function h_dynamics_x!(h_dyn_x::AbstractVector, U::AbstractMatrix, X_k::AbstractMatrix, theta::AbstractArray, V_k::AbstractMatrix)
        for t in 1:H-1
            h_dyn_x[(t-1)*n_x+1:t*n_x] .= f_theta(theta, X_k[:, t], U[:, t]) .+ V_k[:, t] .- X_k[:, t+1]
        end
        return h_dyn_x
    end

    # Helper function that returns the constraint vector containing the dynamics constraints for the states over the whole horizon H for one scenario with a vector valued input z = [vec(U); vec(X^[k]); vec(Y^[k])].
    function h_dynamics_x!(h_dyn_x::AbstractVector, z_scenario::AbstractVector, theta::AbstractArray, V_k::AbstractMatrix)
        U, X_k = unpack_z_scenario(z_scenario, n_u, n_x, n_y, H, J_u)[1:2]
        return h_dynamics_x!(h_dyn_x, U, X_k, theta, V_k)
    end

    # Helper function that evaluates the dynamics constraints for the outputs over the whole horizon H for one scenario.
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

    # Helper function that evaluates constraints for a single scenario.
    function h_scenario!(h::AbstractVector, U::AbstractMatrix, X_k::AbstractMatrix, Y_k::AbstractMatrix)
        h .= vec(h_scenario(U, X_k, Y_k))
        return h
    end

    function h_scenario!(h::AbstractVector, z_scenario::AbstractVector)
        U, X_k, Y_k = unpack_z_scenario(z_scenario, n_u, n_x, n_y, H, J_u)[1:3]
        return h_scenario!(h, U, X_k, Y_k)
    end

    # Helper function that evaluates h_u.
    function h_u!(h::AbstractVector, U::AbstractMatrix)
        h .= vec(h_u(U))
        return h
    end

    function h_u!(h::AbstractVector, U_vec::AbstractVector)
        U = @views reshape(U_vec, n_u, H)
        return h_u!(h, U)
    end

    if !J_u
        function h_J_max!(h::AbstractVector, U::AbstractMatrix, X_k::AbstractMatrix, Y_k::AbstractMatrix, J_max::Union{<:Number,AbstractVector{<:Number}})
            h .= J(U, X_k, Y_k) .- J_max
            return h
        end

        function h_J_max!(h::AbstractVector, z_scenario::AbstractVector)
            U, X_k, Y_k, J_max = unpack_z_scenario(z_scenario, n_u, n_x, n_y, H, J_u)
            return h_J_max!(h, U, X_k, Y_k, J_max)
        end

    else
        # If J_u is true, the J_max constraint is not used.
        h_J_max! = nothing
    end

    if build_hessian
        # Helper function that evaluates the local Lagrangian of the dynamics constraints for the states over the whole horizon H for one scenario.
        function lagrangian_dynamics_x(lambda::AbstractVector, z_scenario::AbstractVector, theta::AbstractArray, V_k::AbstractMatrix)
            h_dyn_x = Vector{eltype(z_scenario)}(undef, n_x * (H - 1))
            h_dynamics_x!(h_dyn_x, z_scenario, theta, V_k)
            return dot(lambda, h_dyn_x)
        end

        # Helper function that evaluates the local Lagrangian of the dynamics constraints for the outputs over the whole horizon H for one scenario.
        function lagrangian_dynamics_y(lambda::AbstractVector, z_scenario::AbstractVector, theta::AbstractArray, W_k::AbstractMatrix)
            h_dyn_y = Vector{eltype(z_scenario)}(undef, n_y * H)
            h_dynamics_y!(h_dyn_y, z_scenario, theta, W_k)
            return dot(lambda, h_dyn_y)
        end

        # Helper function that evaluates the local Lagrangian of the scenario constraints for one scenario.
        function lagrangian_scenario(lambda::AbstractVector, z_scenario::AbstractVector)
            h = Vector{eltype(z_scenario)}(undef, length(z_scenario))
            h_scenario!(h, z_scenario)
            return dot(lambda, h)
        end

        # Helper function that evaluates the local Lagrangian of h_u.
        function lagrangian_u(lambda::AbstractVector, U_vec::AbstractVector)
            h = Vector{eltype(U_vec)}(undef, length(U_vec))
            h_u!(h, U_vec)
            return dot(lambda, h)
        end

        if J_u
            # Helper function that evaluates the local Lagrangian of the epigraph constraint J_max.
            function lagrangian_J_max(lambda::AbstractVector, z_scenario::AbstractVector)
                h = Vector{eltype(z_scenario)}(undef, length(z_scenario))
                h_J_max!(h, z_scenario)
                return dot(lambda, h)
            end
        else
            lagrangian_J_max = nothing
        end
    else
        # If build_hessian is false, the Lagrangian terms are not computed.
        lagrangian_dynamics_x = nothing
        lagrangian_dynamics_y = nothing
        lagrangian_scenario = nothing
        lagrangian_u = nothing
        lagrangian_J_max = nothing
    end

    return h_dynamics_x!, h_dynamics_y!, h_scenario!, h_u!, h_J_max!, lagrangian_dynamics_x, lagrangian_dynamics_y, lagrangian_scenario, lagrangian_u, lagrangian_J_max
end

# The following function computes the sparsity pattern of the local Jacobian of a constraint h! with respect to the decision variables z.
function compute_Jacobian_sparsity(h!, z::AbstractVector, h_loc::AbstractVector)
    # Evaluate the sparsity pattern of the local Jacobian.
    local_Jacobian_sparsity = Symbolics.jacobian_sparsity(h!, h_loc, z)

    # Get matrix coloring.
    local_Jacobian_colors = SparseDiffTools.matrix_colors(local_Jacobian_sparsity)

    # Find the non-zero entries in the sparsity pattern of the dynamics constraints.
    sparsity_local_Jacobian_rows, sparsity_local_Jacobian_columns, _ = findnz(local_Jacobian_sparsity)

    return local_Jacobian_sparsity, local_Jacobian_colors, sparsity_local_Jacobian_rows, sparsity_local_Jacobian_columns
end

# This function expands the sparsity pattern of the local Jacobian of a constraint with respect to a single scenario to the global index space.
# The local Jacobian contains only the considered constraints and is defined with respect to the vector z_scenario, which contains the inputs U, states X_k, outputs Y_k, and (optionally) J_max for scenario k.
# The global Jacobian contains all constraints and is defined with respect to the flat decision vector z, which contains the inputs U, states X of all scenarios, outputs Y of all scenarios, and (optionally) J_max.
# The global indices of nonzero elements are added to the sparsity_global_Jacobian_rows and sparsity_global_Jacobian_columns vectors.
# The input indices_h_global[k] contains the indices of the constraints belonging to scenario k in the global constraint vector h(z).
# The return value nzvals_Jacobian_ranges[k] is the range in the vector containing the non-zero entries of the global Jacobian belonging to scenario k.
function expand_local_Jacobian_sparsity_pattern!(sparsity_global_Jacobian_rows::AbstractVector{<:Integer}, sparsity_global_Jacobian_columns::AbstractVector{<:Integer}, sparsity_local_Jacobian_rows::Vector{Int}, sparsity_local_Jacobian_columns::Vector{Int}, indices_h_global::Vector{UnitRange{Int}}, indices_U::UnitRange{Int}, indices_X::Vector{UnitRange{Int}}, indices_Y::Vector{UnitRange{Int}}, index_J_max::Union{UnitRange{Int},Nothing}, n_u::Int, n_x::Int, n_y::Int, H::Int, K::Int)
    nzvals_Jacobian_ranges = Vector{UnitRange{Int}}(undef, K)
    for k in 1:K
        row_offset = first(indices_h_global[k]) - 1 # offset of the first entry of the constraint in the global constraint vector h(z)
        column_offset_X = first(indices_X[k]) - 1 # offset of the first entry of the state x_{1:H}^{[k]} of scenario k in the global variable vector z
        column_offset_Y = first(indices_Y[k]) - 1 # offset of the first entry of the output y_{1:H}^{[k]} of scenario k in the global variable vector z

        start = length(sparsity_global_Jacobian_rows) + 1 # start index of the non-zero entries of the Jacobian of scenario k in the vector containing the non-zero entries of the global Jacobian

        for (r, c) in zip(sparsity_local_Jacobian_rows, sparsity_local_Jacobian_columns)
            # Translate the local row and column indices to the global index space.
            global_row = row_offset + r
            global_column = translate_local_index_to_global(c, column_offset_X, column_offset_Y, n_u, n_x, n_y, H, indices_U, index_J_max)

            # Add the global row and column indices.
            push!(sparsity_global_Jacobian_rows, global_row)
            push!(sparsity_global_Jacobian_columns, global_column)
        end

        # Get the range of non-zero entries of the Jacobian for scenario k in the vector containing the non-zero entries of the global Jacobian.
        stop = length(sparsity_global_Jacobian_rows)
        nzvals_Jacobian_ranges[k] = start:stop
    end
    return nzvals_Jacobian_ranges, sparsity_global_Jacobian_rows, sparsity_global_Jacobian_columns
end

# The following function computes the sparsity pattern of a constraint for a single scenario with respect to the decision variables of that scenario z_scenario.
# This local sparsity pattern is then expanded to the global index space of the global decision vector z.
# The corresponding indices of nonzero elements are added to the sparsity_global_Jacobian_rows and sparsity_global_Jacobian_columns vectors.
# Then a cache for the automatic differentiation of this constraint is built.
# The input indices_h_global[k] contains the indices of the constraint corresponding to scenario k in the global constraint vector h(z).
function setup_sparse_Jacobian_cache!(sparsity_global_Jacobian_rows::AbstractVector{<:Integer}, sparsity_global_Jacobian_columns::AbstractVector{<:Integer}, h!::Function, h_loc::AbstractVector, z_scenario::AbstractVector, indices_h_global::Vector{UnitRange{Int}}, indices_U::UnitRange{Int}, indices_X::Vector{UnitRange{Int}}, indices_Y::Vector{UnitRange{Int}}, index_J_max::Union{UnitRange{Int},Nothing}, n_u::Int, n_x::Int, n_y::Int, H::Int, K::Int)
    # Evaluate the sparsity pattern of the local Jacobian.
    local_Jacobian_sparsity, local_Jacobian_colors, sparsity_local_Jacobian_rows, sparsity_local_Jacobian_columns = compute_Jacobian_sparsity(h!, z_scenario, h_loc)

    # Expand the local sparsity pattern of the dynamics constraints to the global index space.
    nzrange_Jacobian_h = expand_local_Jacobian_sparsity_pattern!(sparsity_global_Jacobian_rows, sparsity_global_Jacobian_columns, sparsity_local_Jacobian_rows, sparsity_local_Jacobian_columns, indices_h_global, indices_U, indices_X, indices_Y, index_J_max, n_u, n_x, n_y, H, K)[1]

    # Build per-thread caches for the automatic differentiation of the dynamics constraints.
    jacobian_cache = SparseDiffTools.ForwardColorJacCache(h!, z_scenario, nothing; dx=h_loc, colorvec=local_Jacobian_colors, sparsity=local_Jacobian_sparsity)

    cache_Jacobian_h = ADCache(jacobian_cache, Float64.(local_Jacobian_sparsity))
    return nzrange_Jacobian_h, cache_Jacobian_h
end

# The following function computes the sparsity pattern of the Hessian of a function (e.g., the local Lagrangian) with respect to the decision variables z.
function compute_Hessian_sparsity(f::Function, z::AbstractVector)
    # Evaluate the sparsity pattern of the local Hessian.
    local_Hessian_sparsity = Symbolics.hessian_sparsity(f, z)

    # Get matrix coloring.
    local_Hessian_colors = SparseDiffTools.matrix_colors(local_Hessian_sparsity)

    # Find the non-zero entries in the sparsity pattern of the dynamics constraints.
    sparsity_local_Hessian_rows, sparsity_local_Hessian_columns, _ = findnz(local_Hessian_sparsity)

    # Keep only the upper‑triangular entries.
    keep = sparsity_local_Hessian_rows .<= sparsity_local_Hessian_columns
    sparsity_local_Hessian_rows = sparsity_local_Hessian_rows[keep]
    sparsity_local_Hessian_columns = sparsity_local_Hessian_columns[keep]

    return local_Hessian_sparsity, local_Hessian_colors, sparsity_local_Hessian_rows, sparsity_local_Hessian_columns
end

# This function expands the sparsity pattern of the Hessian of a local Lagrangian with respect to a single scenario to the global index space.
# The local Lagrangian is the product of constraints belonging to a single scenario and corresponding multipliers lambda.
# Note that the local Lagrangian is a function of the decision variables z only. This function expects that the the passed function `lagrangian` already contains the Lagrange multipliers.
# The Hessian of the local Lagrangian is defined with respect to the vector z_scenario, which contains the inputs U, states X_k, outputs Y_k, and (optionally) J_max for scenario k.
# The global Hessian contains all constraints and is defined with respect to the flat decision vector z, which contains the inputs U, states X of all scenarios, outputs Y of all scenarios, and (optionally) J_max.
# The global indices of nonzero elements are added to the sparsity_global_Hessian_rows and sparsity_global_Hessian_columns vectors.
# The return value nzvals_Hessian_ranges[k] are the ranges of the Hessian of the local Lagrangian for scenario k in the vector containing the non-zero entries of the global Hessian of the Lagrangian.
function expand_local_Hessian_sparsity!(sparsity_global_Hessian_rows::AbstractVector{<:Integer}, sparsity_global_Hessian_columns::AbstractVector{<:Integer}, sparsity_local_Hessian_rows::Vector{Int}, sparsity_local_Hessian_columns::Vector{Int}, indices_U::UnitRange{Int}, indices_X::Vector{UnitRange{Int}}, indices_Y::Vector{UnitRange{Int}}, index_J_max::Union{UnitRange{Int},Nothing}, n_u::Int, n_x::Int, n_y::Int, H::Int, K::Int)
    nzvals_Hessian_ranges = Vector{UnitRange{Int}}(undef, K)
    for k in 1:K
        offset_X = first(indices_X[k]) - 1 # offset of the first entry of the state x_{1:H}^{[k]} of scenario k in the global variable vector z
        offset_Y = first(indices_Y[k]) - 1 # offset of the first entry of the output y_{1:H}^{[k]} of scenario k in the global variable vector z

        start = length(sparsity_global_Hessian_rows) + 1 # start index of the non-zero entries of the Hessian of scenario k in the vector containing the non-zero entries of the global Hessian

        for (r, c) in zip(sparsity_local_Hessian_rows, sparsity_local_Hessian_columns)
            # Translate the local row and column indices to the global index space.
            global_row = translate_local_index_to_global(r, offset_X, offset_Y, n_u, n_x, n_y, H, indices_U, index_J_max)
            global_column = translate_local_index_to_global(c, offset_X, offset_Y, n_u, n_x, n_y, H, indices_U, index_J_max)

            # Add the global row and column indices.
            push!(sparsity_global_Hessian_rows, global_row)
            push!(sparsity_global_Hessian_columns, global_column)
        end

        # Get the range of non-zero entries of the Hessian for scenario k in the vector containing the non-zero entries of the global Hessian.
        stop = length(sparsity_global_Hessian_rows)
        nzvals_Hessian_ranges[k] = start:stop
    end
    return nzvals_Hessian_ranges, sparsity_global_Hessian_rows, sparsity_global_Hessian_columns
end

# The following function computes the sparsity pattern of the Hessian of a local Lagrangian with respect to the decision variables of that scenario.
# This local sparsity pattern is then expanded to the global index space of the global decision vector z.
# The corresponding indices of nonzero elements are added to the sparsity_global_Hessian_rows and sparsity_global_Hessian_columns vectors.
# Then a cache for the automatic differentiation of this local Lagrangian is built.
function setup_sparse_Hessian_cache!(sparsity_global_Hessian_rows::AbstractVector{<:Integer}, sparsity_global_Hessian_columns::AbstractVector{<:Integer}, lagrangian::Function, h_loc::AbstractVector, z_scenario::AbstractVector, indices_U::UnitRange{Int}, indices_X::Vector{UnitRange{Int}}, indices_Y::Vector{UnitRange{Int}}, index_J_max::Union{UnitRange{Int},Nothing}, n_u::Int, n_x::Int, n_y::Int, H::Int, K::Int)
    # Evaluate the sparsity pattern of the local Jacobian.
    local_Hessian_sparsity, local_Hessian_colors, sparsity_local_Hessian_rows, sparsity_local_Hessian_columns = compute_Hessian_sparsity(lagrangian, z_scenario)

    # Expand the local sparsity pattern of the dynamics constraints to the global index space.
    nzrange_Hessian_L = expand_local_Hessian_sparsity!(sparsity_global_Hessian_rows, sparsity_global_Hessian_columns, sparsity_local_Hessian_rows, sparsity_local_Hessian_columns, indices_U, indices_X, indices_Y, index_J_max, n_u, n_x, n_y, H, K)[1]

    # Build per-thread caches for the automatic differentiation of the dynamics constraints.
    hessian_cache = ForwardColorHesCache(lagrangian, z_scenario, local_Hessian_colors, local_Hessian_sparsity)

    cache_Hessian_L = ADCache(hessian_cache, Float64.(local_Hessian_sparsity))
    return nzrange_Hessian_L, cache_Hessian_L
end