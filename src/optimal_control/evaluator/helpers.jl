# This file contains definitions and helper functions that are called in the constructor of the PMCMC_OCP_Evaluator.
function build_global_helpers(h_u, J, options::OCPOptions, dimensions::OCPDimensions, context::GlobalContext)
    # Helper function that evaluates h_u.
    function h_u!(h::AbstractVector, U::AbstractMatrix)
        h .= vec(h_u(U))
        return h
    end

    function h_u!(h::AbstractVector, U_vec::AbstractVector)
        U = @views reshape(U_vec, dimensions.n_u, dimensions.H)
        return h_u!(h, U)
    end

    if !options.J_u
        eval_J_u = nothing
    else
        function eval_J_u(U_vec::AbstractVector)
            U = @views reshape(U_vec, n_u, H)
            return J(U)
        end
    end

    if options.build_Hessian
        # Helper function that evaluates the local Lagrangian of h_u.
        function lagrangian_u(U_vec::AbstractVector)
            h = Vector{eltype(U_vec)}(undef, dimensions.n_h_u)
            h_u!(h, U_vec)
            return dot(context.lambda_h_u, h)
        end

    else
        lagrangian_u = nothing
    end

    global_helpers = GlobalHelpers(h_u!, eval_J_u, lagrangian_u)
    return global_helpers
end

# The following function builds the helper functions that evaluate the constraints 
function build_thread_helpers(f_theta, g_theta, h_scenario, J, options::OCPOptions, dimensions::OCPDimensions, context::ThreadContext)
    # Helper function that evaluates the dynamics constraints for the states over the whole horizon H for one scenario.
    function h_dynamics_x!(h_dyn_x::AbstractVector, U::AbstractMatrix, X_k::AbstractMatrix)
        for t in 1:dimensions.H-1
            h_dyn_x[(t-1)*dimensions.n_x+1:t*dimensions.n_x] .= f_theta(context.theta, X_k[:, t], U[:, t]) .+ context.V_k[:, t] .- X_k[:, t+1]
        end
        return h_dyn_x
    end

    # Helper function that returns the constraint vector containing the dynamics constraints for the states over the whole horizon H for one scenario with a vector valued input z = [vec(U); vec(X^[k]); vec(Y^[k])].
    function h_dynamics_x!(h_dyn_x::AbstractVector, z_scenario::AbstractVector)
        U, X_k = unpack_z_scenario(z_scenario, options, dimensions)[1:2]
        return h_dynamics_x!(h_dyn_x, U, X_k)
    end

    # Helper function that evaluates the dynamics constraints for the outputs over the whole horizon H for one scenario.
    function h_dynamics_y!(h_dyn_y::AbstractVector, U::AbstractMatrix, X_k::AbstractMatrix, Y_k::AbstractMatrix)
        for t in 1:dimensions.H
            h_dyn_y[(t-1)*dimensions.n_y+1:t*dimensions.n_y] .= g_theta(context.theta, X_k[:, t], U[:, t]) .+ context.W_k[:, t] .- Y_k[:, t]
        end
        return h_dyn_y
    end

    function h_dynamics_y!(h_dyn_y::AbstractVector, z_scenario::AbstractVector)
        U, X_k, Y_k = unpack_z_scenario(z_scenario, options, dimensions)[1:3]
        return h_dynamics_y!(h_dyn_y, U, X_k, Y_k)
    end

    # Helper function that evaluates constraints for a single scenario.
    function h_scenario!(h::AbstractVector, U::AbstractMatrix, X_k::AbstractMatrix, Y_k::AbstractMatrix)
        h .= vec(h_scenario(U, X_k, Y_k))
        return h
    end

    function h_scenario!(h::AbstractVector, z_scenario::AbstractVector)
        U, X_k, Y_k = unpack_z_scenario(z_scenario, options, dimensions)[1:3]
        return h_scenario!(h, U, X_k, Y_k)
    end

    if !options.J_u
        function h_J_max!(h::AbstractVector, U::AbstractMatrix, X_k::AbstractMatrix, Y_k::AbstractMatrix, J_max::Union{<:Number,AbstractVector{<:Number}})
            h .= J(U, X_k, Y_k) .- J_max
            return h
        end

        function h_J_max!(h::AbstractVector, z_scenario::AbstractVector)
            U, X_k, Y_k, J_max = unpack_z_scenario(z_scenario, options, dimensions)
            return h_J_max!(h, U, X_k, Y_k, J_max)
        end
    else
        # If J_u is true, the J_max constraint is not used.
        h_J_max! = nothing
    end

    if options.build_Hessian
        # Helper function that evaluates the local Lagrangian of the dynamics constraints for the states over the whole horizon H for one scenario.
        function lagrangian_dynamics_x(z_scenario::AbstractVector)
            h_dyn_x = Vector{eltype(z_scenario)}(undef, dimensions.n_x * (dimensions.H - 1))
            h_dynamics_x!(h_dyn_x, z_scenario)
            return dot(context.lambda_h_dynamics_x, h_dyn_x)
        end

        # Helper function that evaluates the local Lagrangian of the dynamics constraints for the outputs over the whole horizon H for one scenario.
        function lagrangian_dynamics_y(z_scenario::AbstractVector)
            h_dyn_y = Vector{eltype(z_scenario)}(undef, dimensions.n_y * dimensions.H)
            h_dynamics_y!(h_dyn_y, z_scenario)
            return dot(context.lambda_h_dynamics_y, h_dyn_y)
        end

        # Helper function that evaluates the local Lagrangian of the scenario constraints for one scenario.
        function lagrangian_scenario(z_scenario::AbstractVector)
            h = Vector{eltype(z_scenario)}(undef, length(dimensions.n_h_scenario))
            h_scenario!(h, z_scenario)
            return dot(context.lambda_h_scenario, h)
        end

        if !options.J_u
            # Helper function that evaluates the local Lagrangian of the epigraph constraint J_max.
            function lagrangian_J_max(z_scenario::AbstractVector)
                h = Vector{eltype(z_scenario)}(undef, 1)
                h_J_max!(h, z_scenario)
                return dot(context.lambda_h_J_max, h)
            end
        else
            lagrangian_J_max = nothing
        end
    else
        # If options.build_Hessian is false, the Lagrangian terms are not computed.
        lagrangian_dynamics_x = nothing
        lagrangian_dynamics_y = nothing
        lagrangian_scenario = nothing
        lagrangian_J_max = nothing
    end

    thread_helpers = ThreadHelpers(h_dynamics_x!, h_dynamics_y!, h_scenario!, h_J_max!, lagrangian_dynamics_x, lagrangian_dynamics_y, lagrangian_scenario, lagrangian_J_max)

    return thread_helpers
end