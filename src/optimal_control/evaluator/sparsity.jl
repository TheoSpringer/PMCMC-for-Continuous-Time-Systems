# Helper function that returns the type of coloring that should be used for the Jacobian.
# Column coloring is used for forward mode automatic differentiation.
# Row coloring is used for reverse mode automatic differentiation.
function coloring_partition(backend::ADTypes.AbstractADType)
    mode = ADTypes.mode(backend)
    if mode isa ADTypes.ForwardMode
        return :column
    elseif mode isa ADTypes.ReverseMode
        return :row
    else
        return :column # default fallback
    end
end

# The following type and functions are required to compute the coloring of the Hessian of the Lagrangian once and reuse it later.
# For the Jacobian SparseMatrixColorings.ConstantColoringAlgorithm is used.
# However, the coloring problem for the Hessian of the Lagrangian is symmetric and thus SparseMatrixColorings.ConstantColoringAlgorithm cannot be used.
# We instead define our own type and functions to return the coloring result.
struct ConstantSymmetricColoringAlgorithm{M,R} <: SparseMatrixColorings.AbstractColoringAlgorithm
    template::M
    result::R
end

function ConstantSymmetricColoringAlgorithm(hessian_sparsity::AbstractMatrix; algorithm=SparseMatrixColorings.GreedyColoringAlgorithm())
    if !issymmetric(hessian_sparsity)
        hessian_sparsity = hessian_sparsity .| hessian_sparsity'
    end
    result = SparseMatrixColorings.symmetric_matrix_colors(hessian_sparsity; algorithm)
    return ConstantSymmetricColoringAlgorithm(hessian_sparsity, result)
end

function SparseMatrixColorings.coloring(A, problem::SparseMatrixColorings.ColoringProblem{:symmetric,:direct}, algorithm::ConstantSymmetricColoringAlgorithm; kwargs...)
    if size(A) != size(algorithm.template)
        error("ConstantSymmetricColoring: size mismatch. Got $(size(A)), expected $(size(algorithm.template)).")
    end
    return algorithm.result
end

# The following function computes the sparsity pattern of the Jacobian of a constraint h! with respect to the decision variables z.
function compute_Jacobian_sparsity(h!, z::AbstractVector, h_loc::AbstractVector, options::OCPOptions)
    Jacobian_sparsity = ADTypes.jacobian_sparsity(h!, h_loc, z, options.sparsity_detector)

    # Get matrix coloring.
    if coloring_partition(options.dense_forward_backend) == :column
        Jacobian_coloring = ADTypes.column_coloring(Jacobian_sparsity, options.coloring_algorithm)
    else
        Jacobian_coloring = ADTypes.row_coloring(Jacobian_sparsity, options.coloring_algorithm)
    end

    # Find the non-zero entries in the sparsity pattern.
    sparsity_Jacobian_rows, sparsity_Jacobian_columns, _ = findnz(Jacobian_sparsity)

    return Jacobian_sparsity, Jacobian_coloring, sparsity_Jacobian_rows, sparsity_Jacobian_columns
end

# This function expands the sparsity pattern of the local Jacobian of a constraint with respect to a single scenario to the global index space.
# The local Jacobian contains only the considered constraints and is defined with respect to the vector z_scenario, which contains the inputs U, states X_k, outputs Y_k, and (optionally) J_max for scenario k.
# The global Jacobian contains all constraints and is defined with respect to the flat decision vector z, which contains the inputs U, states X of all scenarios, outputs Y of all scenarios, and (optionally) J_max.
# The global indices of nonzero elements are added to the sparsity_global_Jacobian_rows and sparsity_global_Jacobian_columns vectors.
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
# Then a backend for the automatic differentiation of this constraint block is built.
function register_local_Jacobian_sparsity!(sparsity_global_Jacobian_rows::AbstractVector{<:Integer}, sparsity_global_Jacobian_columns::AbstractVector{<:Integer}, h!::Function, h_loc::AbstractVector, z_scenario::AbstractVector, indices_h_global::Vector{UnitRange{Int}}, options::OCPOptions, dimensions::OCPDimensions, indices::OCPIndices)
    # Evaluate the sparsity pattern of the local Jacobian.
    local_Jacobian_sparsity, local_Jacobian_colors, sparsity_local_Jacobian_rows, sparsity_local_Jacobian_columns = compute_Jacobian_sparsity(h!, z_scenario, h_loc, options)

    # Expand the local sparsity pattern of the dynamics constraints to the global index space.
    nzrange_Jacobian_h = expand_local_Jacobian_sparsity_pattern!(sparsity_global_Jacobian_rows, sparsity_global_Jacobian_columns, sparsity_local_Jacobian_rows, sparsity_local_Jacobian_columns, indices_h_global, options, dimensions, indices)[1]

    # Create a sparsity detector and a coloring algorithm that return the pre-computed pattern/coloring.
    constant_sparsity_detector = ADTypes.KnownJacobianSparsityDetector(local_Jacobian_sparsity)
    constant_coloring_algorithm = SparseMatrixColorings.ConstantColoringAlgorithm(local_Jacobian_sparsity, local_Jacobian_colors; partition=coloring_partition(options.dense_forward_backend))

    backend = DifferentiationInterface.AutoSparse(options.dense_forward_backend, constant_sparsity_detector, constant_coloring_algorithm)

    return nzrange_Jacobian_h, backend
end

# The following function computes the sparsity pattern of the Hessian of the function f with respect to the decision variables z.
function compute_Hessian_sparsity(f::Function, z::AbstractVector, options::OCPOptions)
    Hessian_sparsity = ADTypes.hessian_sparsity(f, z, options.sparsity_detector)

    # Get matrix coloring.
    Hessian_coloring = ADTypes.symmetric_coloring(Hessian_sparsity, options.coloring_algorithm)

    # Find the non-zero entries in the sparsity pattern.
    sparsity_Hessian_rows, sparsity_Hessian_columns, _ = findnz(Hessian_sparsity)

    return Hessian_sparsity, Hessian_coloring, sparsity_Hessian_rows, sparsity_Hessian_columns
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
# Then a backend for the automatic differentiation of this local Lagrangian is built.
function register_local_Hessian_sparsity!(sparsity_global_Hessian_rows::AbstractVector{<:Integer}, sparsity_global_Hessian_columns::AbstractVector{<:Integer}, lagrangian::Function, z_scenario::AbstractVector, options::OCPOptions, dimensions::OCPDimensions, indices::OCPIndices)
    # Evaluate the sparsity pattern of the local Jacobian.
    local_Hessian_sparsity, local_Hessian_colors, sparsity_local_Hessian_rows, sparsity_local_Hessian_columns = compute_Hessian_sparsity(lagrangian, z_scenario, options)

    # Expand the local sparsity pattern of the dynamics constraints to the global index space.
    nzrange_Hessian_L = expand_local_Hessian_sparsity!(sparsity_global_Hessian_rows, sparsity_global_Hessian_columns, sparsity_local_Hessian_rows, sparsity_local_Hessian_columns, options, dimensions, indices)[1]

    # Create a sparsity detector and a coloring algorithm that return the pre-computed pattern/coloring.
    constant_sparsity_detector = ADTypes.KnownHessianSparsityDetector(local_Hessian_sparsity)
    constant_coloring_algorithm = SparseMatrixColorings.ConstantColoringAlgorithm(local_Hessian_sparsity, local_Hessian_colors; partition=:row)

    backend = DifferentiationInterface.AutoSparse(options.dense_second_order_backend, constant_sparsity_detector, constant_coloring_algorithm)

    return nzrange_Hessian_L, backend
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