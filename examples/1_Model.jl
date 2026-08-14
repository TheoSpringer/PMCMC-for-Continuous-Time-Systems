# =============================================================================
# 1_Model.jl  (Glucose–Insulin minimal model, Bergman-type)
# =============================================================================

using LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
# System model container (keep your existing interface)
# ─────────────────────────────────────────────────────────────────────────────
struct SystemModel
    n_x::Int
    n_u::Int
    n_y::Int
    dynamics::Function
    output::Function
    params::Any
end

function _infer_dims(dynamics::Function, output::Function, params; max_x=16, max_u=16)
    for nx in 1:max_x, nu in 1:max_u
        xprobe = zeros(nx)
        uprobe = zeros(nu)
        try
            dx = dynamics(xprobe, uprobe, params, 0.0)
            y  = output(xprobe, uprobe, params, 0.0)
            if isa(dx, AbstractVector) && isa(y, AbstractVector) && length(dx) == nx
                return (nx, nu, length(y))
            end
        catch
            # try next
        end
    end
    error("Could not infer (n_x, n_u, n_y). Provide x0 and u0 explicitly.")
end

function SystemModel(; dynamics::Function,
                      output::Function,
                      params=nothing,
                      x0::Union{Nothing,AbstractVector}=nothing,
                      u0::Union{Nothing,AbstractVector}=nothing)
    if x0 === nothing || u0 === nothing
        nx, nu, ny = _infer_dims(dynamics, output, params)
        return SystemModel(nx, nu, ny, dynamics, output, params)
    else
        dx = dynamics(x0, u0, params, 0.0)
        y  = output(x0, u0, params, 0.0)
        return SystemModel(length(x0), length(u0), length(y), dynamics, output, params)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Meal disturbance schedule (time in minutes; t=0 corresponds to 6pm)
# ─────────────────────────────────────────────────────────────────────────────
struct Meal
    t_meal::Float64   # min
    size::Float64     # mg/dL (lumped)
end

const meals = [
    # -------------------------
    # EXTRA meals in TRAINING window [-36h, 0]
    # (t=0 is 6pm; all times in minutes)
    # -------------------------
    Meal(-34.0 * 60.0, 60.0),   # (training) yesterday breakfast 8am
    Meal(-29.0 * 60.0, 90.0),   # (training) yesterday lunch 1pm
    Meal(-23.0 * 60.0, 80.0),   # (training) yesterday dinner 7pm

    Meal(-19.0 * 60.0, 25.0),   # (training) night snack 11pm
    Meal(-17.0 * 60.0, 20.0),   # (training) night snack 1am

    # -------------------------
    # YOUR ORIGINAL meals (UNCHANGED)
    # -------------------------
    Meal(-10.0 * 60.0, 60.0),   # breakfast 8am
    Meal(-5.0  * 60.0, 90.0),   # lunch     1pm

    Meal(-2.0  * 60.0, 25.0),   # (training) day snack 4pm

    # -------------------------
    # PREDICTED meal (1h after training ends)
    # Training ends at t=0 (hour 36). Hour 37 => t=+1h => +60 min
    # YOUR ORIGINAL entry (UNCHANGED)
    # -------------------------
    Meal( 1.0  * 60.0, 80.0),   # dinner    7pm  (predicted, hour 37)
]

"""
    D_t(t; B=0.05)

Known glucose appearance disturbance, evaluated at time t [min].
(Used as an exogenous input in your SDE/EM code via ZOH.)
"""
function D_t(t; B::Float64=0.05)
    D = 0.0
    for m in meals
        if t >= m.t_meal
            Δ = t - m.t_meal
            D += m.size * B * exp(-B * Δ)
        end
    end
    return D
end

# ─────────────────────────────────────────────────────────────────────────────
# Dynamics
# States: x = [G, X, I]
# Inputs: u = [u_insulin, D]   (both treated ZOH between measurement times)
# Parameters: θ = [log(p2), log(p3), log(n)]  (p1, Gb, Ib treated known)
# ─────────────────────────────────────────────────────────────────────────────
const P1_KNOWN  = 0.0
const GB_KNOWN  = 80.0
const IB_KNOWN  = 7.0

dynamics_fun = function(x, u, p, t)
    θ = p.theta
    p2, p3, n = exp.(θ)

    G = x[1]
    X = x[2]
    I = x[3]

    u_ins = u[1]
    D     = u[2]

    dx1 = -P1_KNOWN * (G - GB_KNOWN) - X * G + D
    dx2 = -p2 * X + p3 * (I - IB_KNOWN)
    dx3 = -n  * (I - IB_KNOWN) + u_ins

    return [dx1, dx2, dx3]
end

# Measurement: glucose only
output_fun = function(x, u, p, t)
    return [x[1]]
end

# Reasonable default θ (log-scale)
const theta_default = [
    log(0.015),   # log(p2)
    log(2e-6),    # log(p3)
    log(0.21)     # log(n)
]

model = SystemModel(; dynamics=dynamics_fun,
                    output=output_fun,
                    params=(theta=theta_default,))
