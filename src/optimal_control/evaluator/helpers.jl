# The following function builds the helper functions that evaluate the constraints h_dynamics_x and h_dynamics_y over the whole horizon as well as h_scenario, h_u, and (optionally) h_J_max with input z_scenario.
# The following struct contains the cached data for the automatic differentiation of one constraint.
struct ConstraintCache
    autodiff_cache::Union{SparseDiffTools.ForwardColorJacCache}
    Jacobian::SparseMatrixCSC{Float64}
end

# The following struct contains the cached data for the automatic differentiation of the dynamic, scenario, and epigraph constraints.
struct ThreadCache
    cache_h_dynamics_x::ConstraintCache
    cache_h_dynamics_y::ConstraintCache
    cache_h_scenario::ConstraintCache
    cache_h_J_max::Union{ConstraintCache,Nothing}
end

function build_helpers(f_theta, g_theta, h_scenario, h_u, J, J_u, n_u::Int, n_x::Int, n_y::Int, H::Int)
    # Helper function that evaluates the dynamics constraints over the whole horizon H for one scenario.
    function h_dynamics_x!(h_dyn_x::AbstractVector, U::AbstractMatrix, X_k::AbstractMatrix, theta::AbstractArray, V_k::AbstractMatrix)
        for t in 1:H-1
            h_dyn_x[(t-1)*n_x+1:t*n_x] .= f_theta(theta, X_k[:, t], U[:, t]) .+ V_k[:, t] .- X_k[:, t+1]
        end
        return h_dyn_x
    end

    # Helper function that returns the constraint vector containing the dynamics constraints over the whole horizon H for one scenario with a vector valued input z = [vec(U); vec(X^[k]); vec(Y^[k])].
    function h_dynamics_x!(h_dyn_x::AbstractVector, z_scenario::AbstractVector, theta::AbstractArray, V_k::AbstractMatrix)
        U, X_k = unpack_z_scenario(z_scenario, n_u, n_x, n_y, H, J_u)[1:2]
        return h_dynamics_x!(h_dyn_x, U, X_k, theta, V_k)
    end

    # Helper function that evaluates the measurement constraints over the whole horizon H for one scenario.
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
        h_J_max! = (h, z_scenario) -> nothing
    end

    return h_dynamics_x!, h_dynamics_y!, h_scenario!, h_u!, h_J_max!
end

function compute_jacobian_sparsity(h!, z_scenario::AbstractVector, h_loc::AbstractVector)
    # Evaluate the sparsity pattern of the local Jacobian.
    local_Jacobian_sparsity = Symbolics.jacobian_sparsity(h!, h_loc, z_scenario)

    # Get matrix coloring.
    local_Jacobian_colors = SparseDiffTools.matrix_colors(local_Jacobian_sparsity)

    # Find the non-zero entries in the sparsity pattern of the dynamics constraints.
    sparsity_local_Jacobian_rows, sparsity_local_Jacobian_columns, _ = findnz(local_Jacobian_sparsity)

    return local_Jacobian_sparsity, local_Jacobian_colors, sparsity_local_Jacobian_rows, sparsity_local_Jacobian_columns
end

# The following function computes the sparsity pattern of a constraint for a single scenario with respect to the decision variables of that scenario z_scenario.
# This local sparsity pattern is then expanded to the global index space of the global decision vector z.
# The corresponding indices of nonzero elements are added to the sparsity_Jacobian_rows_global and sparsity_Jacobian_columns_global vectors.
# Then a cache for the automatic differentiation of this constraint is built.
# The input indices_h_global[k] contains the indices of the constraint corresponding to scenario k in the global constraint vector h(z).
function setup_sparse_Jacobian_cache!(sparsity_global_Jacobian_rows::AbstractVector{<:Integer}, sparsity_global_Jacobian_columns::AbstractVector{<:Integer}, h!::Function, h_loc::AbstractVector, z_scenario::AbstractVector, indices_h_global::Vector{UnitRange{Int}}, indices_U::UnitRange{Int}, indices_X::Vector{UnitRange{Int}}, indices_Y::Vector{UnitRange{Int}}, index_J_max::UnitRange{Int}, n_u::Int, n_x::Int, n_y::Int, H::Int, K::Int)
    # Evaluate the sparsity pattern of the local Jacobian.
    local_Jacobian_sparsity, local_Jacobian_colors, sparsity_local_Jacobian_rows, sparsity_local_Jacobian_columns = compute_jacobian_sparsity(h!, z_scenario, h_loc)

    # Expand the local sparsity pattern of the dynamics constraints to the global index space.
    nzrange_h = expand_local_Jacobian_sparsity_pattern!(sparsity_global_Jacobian_rows, sparsity_global_Jacobian_columns, sparsity_local_Jacobian_rows, sparsity_local_Jacobian_columns, indices_h_global, indices_U, indices_X, indices_Y, index_J_max, n_u, n_x, n_y, H, K)[1]

    # Build per-thread caches for the automatic differentiation of the dynamics constraints.
    autodiff_cache = SparseDiffTools.ForwardColorJacCache(h!, z_scenario, nothing; dx=h_loc, colorvec=local_Jacobian_colors, sparsity=local_Jacobian_sparsity)

    cache_h = ConstraintCache(autodiff_cache, Float64.(local_Jacobian_sparsity))
    return nzrange_h, cache_h
end