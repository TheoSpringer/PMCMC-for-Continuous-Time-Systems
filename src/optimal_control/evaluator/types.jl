struct OCPOptions
    build_Hessian::Bool # if true, the Hessian of the Lagrangian is built
    deduplicate_Hessian::Bool # if true, the Hessian pattern is deduplicated
    J_u::Bool # if false, the epigraph constraint is included
    sparsity_detector::ADTypes.AbstractSparsityDetector # the sparsity detector used to compute the sparsity pattern of the Jacobian and the Hessian of the Lagrangian
    coloring_algorithm::ADTypes.AbstractColoringAlgorithm # the coloring algorithm used to compute the coloring of the Jacobian and the Hessian of the Lagrangian
    dense_forward_backend::ADTypes.AbstractADType # the dense forward differentiation backend used to compute the constraint Jacobian
    dense_second_order_backend::DifferentiationInterface.SecondOrder # the dense second order differentiation backend used to compute the Hessian of the Lagrangian
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

# The following struct contains functions to evaluate a block of the constraint vector h(z) belonging to one scenario and the corresponding Lagrangian.
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

# The following struct contains the cached data for the automatic differentiation of the dynamic, scenario, and epigraph constraints and their Lagrangians.
struct ThreadCache
    Jac_h_dynamics_x::DifferentiationInterface.JacobianPrep
    Jac_h_dynamics_y::DifferentiationInterface.JacobianPrep
    Jac_h_scenario::DifferentiationInterface.JacobianPrep
    Jac_h_J_max::Union{DifferentiationInterface.JacobianPrep,Nothing}

    Hes_L_dynamics_x::Union{DifferentiationInterface.HessianPrep,Nothing}
    Hes_L_dynamics_y::Union{DifferentiationInterface.HessianPrep,Nothing}
    Hes_L_scenario::Union{DifferentiationInterface.HessianPrep,Nothing}
    Hes_L_J_max::Union{DifferentiationInterface.HessianPrep,Nothing}
end

# The following struct contains the cached data for the automatic differentiation of the input constraints.
struct GlobalCache
    Jac_h_u::DifferentiationInterface.JacobianPrep

    Hes_L_u::Union{DifferentiationInterface.HessianPrep,Nothing}
    Hes_J_u::Union{DifferentiationInterface.HessianPrep,Nothing}
end