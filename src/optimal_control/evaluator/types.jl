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
    J_u::Bool # if false, the epigraph constraint is included
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

# The following struct contains functions to evaluate a block of the constraint vector h(z) belonging to one scenario.
# In case a cost function that depends only on the inputs (i.e., J(U)) is used, it also contains the objective function.
struct OCPFunctions
    h_dynamics_x!::Function # dynamic constraints for the states
    h_dynamics_y!::Function # dynamic constraints for the outputs
    h_scenario!::Function # scenario constraints
    h_u!::Function # input constraints
    h_J_max!::Union{Function,Nothing} # epigraph constraints (optional)
    eval_J_u::Union{Function,Nothing} # evaluates J(u)
    lagrangian_dynamics_x::Union{Function,Nothing} # Lagrangian of the dynamics constraints for the states
    lagrangian_dynamics_y::Union{Function,Nothing} # Lagrangian of the dynamics constraints for the outputs
    lagrangian_scenario::Union{Function,Nothing} # Lagrangian of the scenario constraints
    lagrangian_u::Union{Function,Nothing} # Lagrangian of the input constraints
    lagrangian_J_max::Union{Function,Nothing} # Lagrangian of the epigraph constraints (optional)
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

# The following struct contains the cached data for the automatic differentiation.
struct ADCache{CacheType,MatrixType}
    autodiff_cache::CacheType
    buffer::MatrixType
end

# The following struct contains the cached data for the automatic differentiation of the dynamic, scenario, and epigraph constraints.
struct ThreadCache
    Jacobian_h_dynamics_x::ADCache
    Jacobian_h_dynamics_y::ADCache
    Jacobian_h_scenario::ADCache
    Jacobian_h_J_max::Union{ADCache,Nothing}

    Hessian_L_dynamics_x::Union{ADCache,Nothing}
    Hessian_L_dynamics_y::Union{ADCache,Nothing}
    Hessian_L_scenario::Union{ADCache,Nothing}
    Hessian_L_J_max::Union{ADCache,Nothing}
end