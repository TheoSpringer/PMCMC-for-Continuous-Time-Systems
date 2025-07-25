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
function expand_local_Jacobian_sparsity_pattern!(sparsity_global_Jacobian_rows::AbstractVector{<:Integer}, sparsity_global_Jacobian_columns::AbstractVector{<:Integer}, sparsity_local_Jacobian_rows::Vector{Int}, sparsity_local_Jacobian_columns::Vector{Int}, indices_h_global::Vector{UnitRange{Int}}, options::OCPOptions, dimensions::OCPDimensions, indices::OCPIndices)
    nzvals_Jacobian_ranges = Vector{UnitRange{Int}}(undef, dimensions.K)
    for k in 1:dimensions.K
        row_offset = first(indices_h_global[k]) - 1 # offset of the first entry of the constraint in the global constraint vector h(z)
        column_offset_X = first(indices.X[k]) - 1 # offset of the first entry of the state x_{1:H}^{[k]} of scenario k in the global variable vector z
        column_offset_Y = first(indices.Y[k]) - 1 # offset of the first entry of the output y_{1:H}^{[k]} of scenario k in the global variable vector z

        start = length(sparsity_global_Jacobian_rows) + 1 # start index of the non-zero entries of the Jacobian of scenario k in the vector containing the non-zero entries of the global Jacobian

        for (r, c) in zip(sparsity_local_Jacobian_rows, sparsity_local_Jacobian_columns)
            # Translate the local row and column indices to the global index space.
            global_row = row_offset + r
            global_column = translate_local_index_to_global(c, column_offset_X, column_offset_Y, options, dimensions, indices)

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
# The input indices_h_global[k] contains the indices of the constraint corresponding to scenario k in the global constraint vector h(z).
function register_local_Jacobian_sparsity!(sparsity_global_Jacobian_rows::AbstractVector{<:Integer}, sparsity_global_Jacobian_columns::AbstractVector{<:Integer}, h!::Function, h_loc::AbstractVector, z_scenario::AbstractVector, indices_h_global::Vector{UnitRange{Int}}, options::OCPOptions, dimensions::OCPDimensions, indices::OCPIndices)
    # Evaluate the sparsity pattern of the local Jacobian.
    local_Jacobian_sparsity, local_Jacobian_colors, sparsity_local_Jacobian_rows, sparsity_local_Jacobian_columns = compute_Jacobian_sparsity(h!, z_scenario, h_loc)

    # Expand the local sparsity pattern of the dynamics constraints to the global index space.
    nzrange_Jacobian_h = expand_local_Jacobian_sparsity_pattern!(sparsity_global_Jacobian_rows, sparsity_global_Jacobian_columns, sparsity_local_Jacobian_rows, sparsity_local_Jacobian_columns, indices_h_global, options, dimensions, indices)[1]

    return nzrange_Jacobian_h, local_Jacobian_colors, local_Jacobian_sparsity
end

# The following function computes the sparsity pattern of the Hessian of a function (e.g., the local Lagrangian) with respect to the decision variables z.
function compute_Hessian_sparsity(f::Function, z::AbstractVector)
    # Evaluate the sparsity pattern of the local Hessian.
    local_Hessian_sparsity = Symbolics.hessian_sparsity(f, z)

    # Keep only the lower triangular part of the Hessian sparsity pattern.
    local_Hessian_sparsity = tril(local_Hessian_sparsity, 0)

    # Get matrix coloring.
    local_Hessian_colors = SparseDiffTools.matrix_colors(local_Hessian_sparsity)

    # Find the non-zero entries in the sparsity pattern of the dynamics constraints.
    sparsity_local_Hessian_rows, sparsity_local_Hessian_columns, _ = findnz(local_Hessian_sparsity)

    return local_Hessian_sparsity, local_Hessian_colors, sparsity_local_Hessian_rows, sparsity_local_Hessian_columns
end

# This function expands the sparsity pattern of the Hessian of a local Lagrangian with respect to a single scenario to the global index space.
# The local Lagrangian is the product of constraints belonging to a single scenario and corresponding multipliers lambda.
# Note that the local Lagrangian is a function of the decision variables z only. This function expects that the the passed function `lagrangian` already contains the Lagrange multipliers.
# The Hessian of the local Lagrangian is defined with respect to the vector z_scenario, which contains the inputs U, states X_k, outputs Y_k, and (optionally) J_max for scenario k.
# The global Hessian contains all constraints and is defined with respect to the flat decision vector z, which contains the inputs U, states X of all scenarios, outputs Y of all scenarios, and (optionally) J_max.
# The global indices of nonzero elements are added to the sparsity_global_Hessian_rows and sparsity_global_Hessian_columns vectors.
# The return value nzvals_Hessian_ranges[k] are the ranges of the Hessian of the local Lagrangian for scenario k in the vector containing the non-zero entries of the global Hessian of the Lagrangian.
function expand_local_Hessian_sparsity!(sparsity_global_Hessian_rows::AbstractVector{<:Integer}, sparsity_global_Hessian_columns::AbstractVector{<:Integer}, sparsity_local_Hessian_rows::Vector{Int}, sparsity_local_Hessian_columns::Vector{Int}, options::OCPOptions, dimensions::OCPDimensions, indices::OCPIndices)
    nzvals_Hessian_ranges = Vector{UnitRange{Int}}(undef, dimensions.K)
    for k in 1:dimensions.K
        offset_X = first(indices.X[k]) - 1 # offset of the first entry of the state x_{1:H}^{[k]} of scenario k in the global variable vector z
        offset_Y = first(indices.Y[k]) - 1 # offset of the first entry of the output y_{1:H}^{[k]} of scenario k in the global variable vector z

        start = length(sparsity_global_Hessian_rows) + 1 # start index of the non-zero entries of the Hessian of scenario k in the vector containing the non-zero entries of the global Hessian

        for (r, c) in zip(sparsity_local_Hessian_rows, sparsity_local_Hessian_columns)
            # Translate the local row and column indices to the global index space.
            global_row = translate_local_index_to_global(r, offset_X, offset_Y, options, dimensions, indices)
            global_column = translate_local_index_to_global(c, offset_X, offset_Y, options, dimensions, indices)

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
function register_local_Hessian_sparsity!(sparsity_global_Hessian_rows::AbstractVector{<:Integer}, sparsity_global_Hessian_columns::AbstractVector{<:Integer}, lagrangian::Function, z_scenario::AbstractVector, options::OCPOptions, dimensions::OCPDimensions, indices::OCPIndices)
    # Evaluate the sparsity pattern of the local Jacobian.
    local_Hessian_sparsity, local_Hessian_colors, sparsity_local_Hessian_rows, sparsity_local_Hessian_columns = compute_Hessian_sparsity(lagrangian, z_scenario)

    # Expand the local sparsity pattern of the dynamics constraints to the global index space.
    nzrange_Hessian_L = expand_local_Hessian_sparsity!(sparsity_global_Hessian_rows, sparsity_global_Hessian_columns, sparsity_local_Hessian_rows, sparsity_local_Hessian_columns, options, dimensions, indices)[1]

    return nzrange_Hessian_L, local_Hessian_colors, local_Hessian_sparsity
end

# The following function deduplicates the non-zero entries of the global Hessian sparsity pattern.
# Generally, duplicates are expected to occur but this is generally not a problem since they are just summed up.
# However, in certain cases (e.g., for checking the structure) it might be necessary to deduplicate the Hessian pattern.
function deduplicate_pattern(pattern::Vector{Tuple{Int,Int}}, nz_values::Vector{Float64})
    @assert length(pattern) == length(nz_values)

    dict = Dict{Tuple{Int,Int},Float64}()
    for (i, key) in enumerate(pattern)
        dict[key] = get(dict, key, 0.0) + nz_values[i]
    end

    deduplicated_pattern = collect(keys(dict))
    deduplicated_nz_values = collect(values(dict))

    return deduplicated_pattern, deduplicated_nz_values
end