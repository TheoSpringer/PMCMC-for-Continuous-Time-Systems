struct OCPOptions
    build_Hessian::Bool # if true, the Hessian of the Lagrangian is built
    deduplicate_Hessian::Bool # if true, the Hessian pattern is deduplicated
    J_u::Bool # if false, the epigraph constraint is included
end

# The following struct contains all dimensions relevant for the optimal control problem.
struct OCPDimensions
    n_u::Int
    n_x::Int
    n_y::Int
    K::Int
    H::Int
    n_z::Int # total number of decision variables
    n_z_scenario::Int # number of decision variables per scenario
    n_h_scenario::Int # number of scenario constraints
    n_h_u::Int # number of input constraints
    n_h::Int # total number of constraints
end

# The following struct contains the indices of the variables in the flat decision vector z and the constraint vector h(z).
struct OCPIndices
    U::UnitRange{Int}
    X::Vector{UnitRange{Int}}
    Y::Vector{UnitRange{Int}}
    J_max::UnitRange{Int}
    h_dynamics_x::Vector{UnitRange{Int}}
    h_dynamics_y::Vector{UnitRange{Int}}
    h_scenario::Vector{UnitRange{Int}}
    h_u::UnitRange{Int}
    h_J_max::Union{Vector{UnitRange{Int}},Nothing}
end

# The following struct contains the data of the optimal control problem.
struct OCPData
    PMCMC_samples::Vector{PMCMC_sample}
    V::Array{Float64,3}
    W::Array{Float64,3}
    X_t::Array{Float64,2}
end

# The following struct contains the ranges corresponding to specific constraints in the vector containing the non-zero entries of the global Jacobian.
struct SparseJacobianNZRanges
    h_dynamics_x::Vector{UnitRange{Int}}
    h_dynamics_y::Vector{UnitRange{Int}}
    h_scenario::Vector{UnitRange{Int}}
    h_u::UnitRange{Int}
    h_J_max::Union{Vector{UnitRange{Int}},Nothing}
end

# The following struct contains the ranges corresponding to specific constraints in the vector containing the non-zero entries of the global Hessian of the Lagrangian.
struct SparseHessianNZRanges
    L_dynamics_x::Vector{UnitRange{Int}}
    L_dynamics_y::Vector{UnitRange{Int}}
    L_scenario::Vector{UnitRange{Int}}
    L_u::UnitRange{Int}
    L_J_max::Union{Vector{UnitRange{Int}},Nothing}
    L_J_u::Union{UnitRange{Int},Nothing}
end

# The following struct contains the context for a single thread.
# The context holds references to the data of a scenario (theta, V, W), 
# the current Lagrange multipliers, and some buffers used to evaluate the Lagrangian.
# During the optimization, these references are mutated in place (their contents are overwritten) 
# so that a single pre-built closure and a single AD cache can be reused without reallocation.
mutable struct ThreadContext{Tθ,TV,TW,Tλ}
    # Data
    theta::AbstractArray{Tθ}
    V_k::AbstractArray{TV}
    W_k::AbstractArray{TW}

    # Lagrange multipliers
    lambda_h_dynamics_x::AbstractVector{Tλ}
    lambda_h_dynamics_y::AbstractVector{Tλ}
    lambda_h_scenario::AbstractVector{Tλ}
    lambda_h_J_max::Union{AbstractVector{Tλ},Nothing}
end

mutable struct GlobalContext{T}
    lambda_h_u::AbstractVector{T}
end

# The following struct contains functions to evaluate the dynamic constraints for the states and outputs, the scenario constraints, and the epigraph constraints for a scenario and the corresponding Lagrangians.
struct ThreadHelpers
    h_dynamics_x!::Function # dynamic constraints for the states
    h_dynamics_y!::Function # dynamic constraints for the outputs
    h_scenario!::Function # scenario constraints
    h_J_max!::Union{Function,Nothing} # epigraph constraints (optional)
    lagrangian_dynamics_x::Union{Function,Nothing} # Lagrangian of the dynamics constraints for the states
    lagrangian_dynamics_y::Union{Function,Nothing} # Lagrangian of the dynamics constraints for the outputs
    lagrangian_scenario::Union{Function,Nothing} # Lagrangian of the scenario constraints
    lagrangian_J_max::Union{Function,Nothing} # Lagrangian of the epigraph constraints (optional)
end

# The following struct contains functions to evaluate the input constraints h(u), the corresponding Lagrangian, and the cost function if it depends only on the inputs u.
struct GlobalHelpers
    h_u!::Function # input constraints
    eval_J_u::Union{Function,Nothing} # evaluates J(u)
    lagrangian_u::Union{Function,Nothing} # Lagrangian of the input constraints
end

# The following struct contains the cached data for the automatic differentiation.
struct ADCache{CacheType,MatrixType}
    autodiff_cache::CacheType
    buffer::MatrixType
end

# The following struct contains the cached data for the automatic differentiation of the dynamic, scenario, and epigraph constraints and their Lagrangians.
struct ThreadCache
    Jacobian_h_dynamics_x::ADCache
    Jacobian_h_dynamics_y::ADCache
    Jacobian_h_scenario::ADCache
    Jacobian_h_J_max::Union{ADCache,Nothing}

    Hessian_L_dynamics_x::Union{ADCache,Nothing}
    Hessian_L_dynamics_y::Union{ADCache,Nothing}
    Hessian_L_scenario::Union{ADCache,Nothing}
    Hessian_L_J_max::Union{ADCache,Nothing}

    helpers::ThreadHelpers
    context::ThreadContext
end

# The following struct contains the cached data for the automatic differentiation of the input constraints.
struct GlobalCache
    Jacobian_h_u::ADCache

    Hessian_L_u::Union{ADCache,Nothing}
    Hessian_J_u::Union{ADCache,Nothing}

    context::GlobalContext
    helpers::GlobalHelpers
end