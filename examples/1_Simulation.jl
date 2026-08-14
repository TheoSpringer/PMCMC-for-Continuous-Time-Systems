using LinearAlgebra 
using Random
using Distributions
using Plots
using Printf
using JLD2
using Statistics: mean, var, median, cov
using ScenarioPMCMC
using ScenarioOCP
using StatsBase: Weights, sample, autocor

const THESIS_TITLE_FSIZE  = 56   # plot title
const THESIS_LABEL_FSIZE  = 44   # x/y labels (guides)
const THESIS_TICK_FSIZE   = 30   # tick labels
const THESIS_LEGEND_FSIZE = 30   # legend text
const THESIS_LW           = 10    # line width
const THESIS_MS           = 15   # marker size
# --- Global plotting defaults (big + crisp) ---
default(
    size=(2400, 1400),    # big canvas → many pixels
    dpi=200,              # high-density rendering
    lw=THESIS_LW,                 # thicker lines
    ms=THESIS_MS,                 # marker size
    legendfontsize=THESIS_LEGEND_FSIZE,
    guidefontsize=THESIS_LABEL_FSIZE,
    tickfontsize=THESIS_TICK_FSIZE
)
# --- bring in your model and helpers (clean separation) ---
gr()
include("1_Model.jl")
include("1_Helpers.jl")
include("1_PF.jl")

# Time PMCMC algorithm.
sampling_timer = time()
const H_TRUTH  = 0.1
Random.seed!(1234)

###############################
# Purpose 
###############################
# We:
# 1) Config / model sizes
# 2) Define priors for x0 and θ.
# 3) Define noise models.
# 4) Inputs and ZOH
# 5) Generate measured IO data at irregular times (TRUE system)
# 6) PMMH wiring (train on measured data)
# 7) staged_PMMH
# 8) Plots 

#=Where can i see the bias?

transition bias - see Particl Filter inside particle_MMH_dt: how process noise is injected (outside the integrator, once per gap)
adding all process noise once per gap (as Q*Δt after the integrator) is not the true SDE solution; process noise should be interleaved with drift along the path. You already provide both styles:
“per sub-step EM” (your em_propagate_gap!), which is the correct SDE semantics.


discretization bias: - see f_theta: one jump over each gap Δt using a numerical ODE solver with finite tolerances
any fixed-step ODE/SDE solver approximates the continuous path; larger gaps Δt make the bias more dangerous. Your notes say this explicitly. 
Your code removes likelihood-level bias via randomized multilevel (Rhee–Glynn), so posterior bias from the integrator is eliminated in expectation.=#

###############################
# 1) Config / model sizes
###############################.
const n_x = model.n_x
const n_u = model.n_u
const n_y = model.n_y
const n_θ = length(model.params.theta)

###############################
# 2) Priors for x0 and θ  (glucose-insulin)
###############################

# True parameters (log-scale) and true initial state
θ_true = [-4.22948, -13.10747, -1.62438] # p2≈0.01456, p3≈2.03e-6, n≈0.1970

x0_true = [75.0, 5e-4, 8.0]

# Prior for x0 ~ independent truncated Normals
const μx   = [80.0, 0.0, 7.0]
const σx   = [10.0, 0.001, 2.0]
const lo_x = [20.0, -Inf, 0.0]
const hi_x = [500.0, Inf, 200.0]

x0_dists = [Truncated(Normal(μx[i], σx[i]), lo_x[i], hi_x[i]) for i in 1:n_x]
sample_x_0()       = [rand(x0_dists[i]) for i in 1:n_x]
sample_x_0(N::Int) = hcat([sample_x_0() for _ in 1:N]...)
logpdf_x_0(x::AbstractVector) = sum(logpdf.(x0_dists, x))
logpdf_x_0(X::AbstractMatrix) = permutedims([sum(logpdf.(x0_dists, view(X, :, i))) for i in 1:size(X, 2)])

# Prior for θ = [log(p2), log(p3), log(n)] ~ truncated Normal (log-normal in physical scale)
const μθ = [-4.26, -13.27, -1.66]
const σθ = [0.75*0.18, 0.75*0.28, 0.75*0.23]  

# Bounds derived from reported ranges with a small margin
const θ_lo = [log(1e-3), log(1e-9), log(0.05)]
const θ_hi = [log(1e-1), log(1e-4), log(1.0)]

const θ_dists = [Truncated(Normal(μθ[i], σθ[i]), θ_lo[i], θ_hi[i]) for i in 1:n_θ]
sample_theta()       = [rand(θ_dists[i]) for i in 1:n_θ]
sample_theta(N::Int) = hcat([sample_theta() for _ in 1:N]...)
logpdf_theta(θ::AbstractVector) = sum(logpdf.(θ_dists, θ))
logpdf_theta(Θ::AbstractMatrix) = permutedims([sum(logpdf.(θ_dists, view(Θ, :, i))) for i in 1:size(Θ, 2)])

###############################
# 3) Noise models
###############################

# Process diffusion Q (keep small; model is essentially ODE with mild process noise)
Q = Diagonal([0.2^2, (1e-4)^2, 0.2^2]) .* 0.7# [G, X, I] units per minute
const G = cholesky(Q).L               # diffusion factor

# Measurement noise: glucose only
const σy_eps = 1.0 # was 1.5
R = Diagonal([σy_eps^2])

sample_v_theta(θ, Np, Δt::Real) = rand(MvNormal(zeros(n_x), Q * max(Δt, eps())), Np)
sample_w_theta(θ, Np) = rand(MvNormal(zeros(n_y), R), Np)

function log_pdf_w_theta(θ, W::AbstractMatrix)
    if ndims(W) == 1
        return -0.5 * (W' * (R \ W))
    else
        return -0.5 .* sum(W .* (R \ W), dims=1)
    end
end

# drift wrapper used by PF.jl
@inline function drift!(b::AbstractVector, x::AbstractVector, u::AbstractVector, θ::AbstractVector)
    bt = model.dynamics(x, u, (theta=θ,), 0.0)
    if bt isa Number
        @inbounds b[1] = float(bt)
    else
        @inbounds @simd for i in eachindex(b)
            b[i] = bt[i]
        end
    end
    return b
end


###############################
# 4) Inputs and ZOH (glucose-insulin)
###############################

T_train = 36 * 60.0     # minutes  (36h training)
T_test  = 12 * 60.0     # minutes  (12h testing)
T_total = T_train + T_test  # = 48h total

# -----------------------------
# Measurement times (grid-aligned, coupling-friendly)
# -----------------------------
const ΔBASE = 1.0    # minutes (keep = finest discretization grid)
const ΔMEAS = 32.0   # minutes (target measurement gap)
# Optional dropout (keep probability). Keep your existing values if already defined elsewhere.
const P_KEEP_TRAIN = 0.7
const P_KEEP_TEST  = 0.7

rng_meas = MersenneTwister(2026)

# Regular 30-min grid, exclude endpoints and exclude t=0 for clean split
t_train = collect((-T_train + ΔMEAS):ΔMEAS:(-ΔMEAS))     # (-36h, 0)
t_test  = collect(ΔMEAS:ΔMEAS:(T_test - ΔMEAS))          # (0, +12h)

# Apply dropout (partial observations) — keep your existing probabilities
t_train = [t for t in t_train if rand(rng_meas) < P_KEEP_TRAIN]
t_test  = [t for t in t_test  if rand(rng_meas) < P_KEEP_TEST ]

# Optional: enforce at least 1 point in each window (still on-grid)
if isempty(t_train); t_train = [-ΔMEAS]; end
if isempty(t_test);  t_test  = [ ΔMEAS]; end

# Final measurement grid (aligned + unique + strictly increasing)
t_meas = sort(unique(vcat(t_train, t_test)))
@assert all(diff(t_meas) .> 0)

# Sanity: everything is exactly on the 1-min base grid
@assert maximum(abs.(t_meas ./ ΔBASE .- round.(t_meas ./ ΔBASE))) < 1e-10



# Split index by TIME (robust after dropout/thinning)
i_split = findlast(<(0.0), t_meas)
i_split = isnothing(i_split) ? 0 : i_split

t_train_sorted = t_meas[1:i_split]
t_test_sorted  = t_meas[i_split+1:end]


# Insulin bolus input u(t) (ZOH at measurement times)
const T_bolus = 60.0
const k_ins   = 0.22

function u_t_bolus(t)
    u = 0.0
    for m in meals
        if m.t_meal <= t && t < m.t_meal + T_bolus
            u += k_ins * m.size
        end
    end
    return u
end

# -----------------------------
# Inputs as a function of time (independent of measurement times)
# -----------------------------
u_of_t(t) = begin
    u_ins = u_t_bolus(t)
    D     = D_t(t)
    return [u_ins, D]
end

# Build U_meas by sampling the input function at measurement times
U_meas = zeros(n_u, length(t_meas))
for k in 1:length(t_meas)
    U_meas[:, k] = u_of_t(t_meas[k])
end

###############################
# 6) PMMH wiring (train on measured data)
###############################
# Measurement/output map (multi)
function g_theta(θ::AbstractVector, X::AbstractMatrix, U::AbstractMatrix)
    Np = size(X, 2)
    Y  = Matrix{Float64}(undef, model.n_y, Np)
    @inbounds for i in 1:Np
        Y[:, i] = model.output(view(X, :, i), view(U, :, i), (theta=θ,), 0.0)
    end
    return Y
end

# Measurement/output map (single)
g_theta(θ::AbstractVector, x::AbstractVector, u::AbstractVector) =
    model.output(x, u, (theta=θ,), 0.0)



# AUGMENT INPUTS FOR PMMH WITH PER-STEP Δt
# Left-constant control for propagation on (t_{k-1}, t_k]:
Δt_cols = [0.0; diff(t_meas)]
U_left  = hcat(U_meas[:, 1], U_meas[:, 1:end-1])
U_aug   = vcat(U_left, reshape(Δt_cols, 1, :))

U_meas_train = @view U_meas[:, 1:i_split]
U_meas_test  = @view U_meas[:, i_split+1:end]
U_aug_train  = @view U_aug[:, 1:i_split]
U_aug_test   = (i_split < size(U_aug,2)) ? (@view U_aug[:, i_split+1:end]) :
                                          zeros(eltype(U_aug), size(U_aug,1), 0)

U_meas_train = @view U_meas[:, 1:i_split]

###############################
# 5) Generate truth and observations
###############################
t_dense, X_dense, Y_dense, X_meas, Y_meas_clean =
    simulate_truth_em(t_meas, U_meas, θ_true, x0_true;
                      dt_truth=0.05, rng=MersenneTwister(2025),
                      add_process_noise=true)

# Clean (noise-free) outputs at measurement times
Y_true = copy(Y_meas_clean)

# Noisy observations (what PMMH conditions on)
Y_obs = copy(Y_true)
Y_obs .+= rand(MvNormal(zeros(n_y), R), size(Y_obs, 2))

# Split into train/test
X_train      = X_meas[:, 1:i_split]
X_test_true  = X_meas[:, i_split+1:end]

Y_train_true = Y_true[:, 1:i_split]
Y_test_true  = Y_true[:, i_split+1:end]

Y_train_obs  = Y_obs[:, 1:i_split]
Y_test_obs   = Y_obs[:, i_split+1:end]
Y_train = Y_train_obs
# PMMH trains on OBSERVATIONS (noisy)
y_training = Y_train_obs

function diagnose_mlpf_noise(u_aug::AbstractMatrix,
                             u_meas::AbstractMatrix,
                             y::AbstractMatrix,
                             n_x::Int,
                             N_pf::Int,
                             sample_x_0::Function,
                             θ::AbstractVector;
                             n_trials::Int = 30,
                             ℓ0::Int = 2,
                             ρ_tail::Float64 = 0.5,
                             ℓ_max::Int = 6,
                             resample_ess_frac::Float64 = 0.8,
                             n_rep_mlpf::Int = 2,
                             seed::Int = 12345)

    rng = MersenneTwister(seed)
    logabs = fill(-Inf, n_trials)
    sgn    = zeros(Float64, n_trials)

    for r in 1:n_trials
        _, _, logabs[r], sgn[r] = mlpf_unbiased_likelihood(
            u_aug, u_meas, y, n_x, N_pf, sample_x_0, θ;
            ℓ0=ℓ0, ρ=ρ_tail, ℓ_max=ℓ_max,
            resample_ess_frac=resample_ess_frac,
            n_rep_mlpf=n_rep_mlpf,
            rng=rng
        )
    end

    finite = isfinite.(logabs)
    n_finite = count(finite)
    if n_finite < max(5, Int(0.8 * n_trials))
        @printf("\n[MLPF diag] WARNING: only %d/%d finite log|Z| draws.\n", n_finite, n_trials)
    end

    meanS   = mean(sgn)
    fracNeg = mean(sgn .< 0)
    essS    = (sum(sgn)^2) / max(sum(sgn.^2), eps())
    vlog    = (n_finite >= 2) ? var(logabs[finite]) : Inf

    @printf("\n[MLPF diag @ fixed θ]\n")
    @printf("  N_pf=%d, ℓ0=%d, ρ=%.3f, ℓ_max=%d, n_rep=%d, ess_frac=%.2f\n",
            N_pf, ℓ0, ρ_tail, ℓ_max, n_rep_mlpf, resample_ess_frac)
    @printf("  Var(log|Ẑ|)=%.3f  (finite draws %d/%d)\n", vlog, n_finite, n_trials)
    @printf("  mean(sign)=%.4f  fracNeg=%.4f  ess_sign=%.1f\n", meanS, fracNeg, essS)
    @printf("  log|Ẑ| range: min=%.2f  median=%.2f  max=%.2f\n",
        minimum(logabs[finite]), median(logabs[finite]), maximum(logabs[finite]))
    return (; vlog, meanS, fracNeg, essS, n_finite)
end

function empirical_cov_theta(PMMH_samples; burn::Int=0)
    K = length(PMMH_samples)
    i0 = max(1, burn+1)
    Θ = hcat(getfield.(PMMH_samples, :theta)...)[:, i0:K]   # d × N
    X = permutedims(Θ)                                      # N × d
    Σ = cov(X; corrected=true)                              # d × d
    return Symmetric(Σ)
end






#=####################################
7) staged_PMMH
#################################### =#
# -------- choose which integrator/likelihood variant to use --------
#const VARIANT  = :mlpf_signed  
#const VARIANT = :em_coarse 
const VARIANT = :em_fine  
const H_COARSE = 16.0#30   
const H_FINE   = 1.0   
# ---- Posterior predictive controls (used by all plots) ----
const PP_INCLUDE_STATE_UNC     = true
const PP_INCLUDE_PROCESS_NOISE = true
const PP_INCLUDE_MEAS_NOISE    = true
const SHOW_TRAIN_BAND = false  # set true if you want train uncertainty too
const PP_R                     = 5      # per-θ replicate draws (state/noise)
# Heavier PF only for *integrator comparison* (not for the full PMMH chain)
const N_PF_COMPARE = 120      # 150–250 is a good range
const REPS_COMPARE = 8       # 20–30 repetitions to stabilize Var[log p̂] estimate

#old Tuning needs to be unfolded
#theta_init = [-4.124843685627391, -13.2275788491581, -1.7667284059340966]
    #proposal_sd =  1.3 .* [0.05, 0.07, 0.12]
    #proposal_cov_init = Diagonal(proposal_sd.^2)
    #K       = 800#2000       # Total Metropolis–Hastings iterations (posterior samples) to draw
    #K_b     = 200#500        # Burn-in samples
    #k_d     = 0#            # Thinning setting
    #N_pf    = 600#900       # Number of particles in the bootstrap filter used to estimate the marginal likelihood 𝑝(𝑦∣𝜃): Larger ⇒ lower variance of log-likelihood (better acceptance, more stable chain) but linearly more compute. Too small ⇒ noisy likelihood ⇒ sticky chain or rejections.
    #T_chunk = 20#10        # T_chunk = cheaper/less noisy early likelihoods ⇒ faster early iterations but you need enough staging to reach the full data.
    #K_stage = 50#15        # Larger K_stage gives more iterations per stage (better local adaptation) but increases wall-time.
    #alpha = 0.35 * (2.38^2) / length(theta_init) #0.05 * (2.38^2) / length(theta_init)        # Global scaling applied to the proposal covariance: 0.2–0.35 acceptance; 0.35 makes it more conservative
    #regularizer = 1e-6


# Current tuning 
theta_init = [-4.316982210209795, -13.295645122830793, -1.599005092242251]
Σ_emp = [
     0.009306155123093341   -0.0002995952619018837  -0.00601562455928151;
    -0.0002995952619018837   0.006063094710706837    0.002181891606613402;
    -0.00601562455928151     0.002181891606613402    0.007275249103170327
]
d = length(theta_init)
regularizer = 1e-6
s2 = 0.6608466666666666 
proposal_cov_init = Symmetric(s2 .* Σ_emp .+ regularizer .* Matrix(I, d, d))
alpha = 1

K       = 1000#3000       # Total Metropolis–Hastings iterations (posterior samples) to draw
K_b     = 200#500        # Burn-in samples
k_d     = 0#            # Thinning setting
N_pf    = 1000       # Number of particles in the bootstrap filter used to estimate the marginal likelihood 𝑝(𝑦∣𝜃): Larger ⇒ lower variance of log-likelihood (better acceptance, more stable chain) but linearly more compute. Too small ⇒ noisy likelihood ⇒ sticky chain or rejections.

T_chunk = 4#10        # T_chunk = cheaper/less noisy early likelihoods ⇒ faster early iterations but you need enough staging to reach the full data.
K_stage = 70#15        # Larger K_stage gives more iterations per stage (better local adaptation) but increases wall-time.

# Diagnose at a sensible θ (start point)
    #@printf("started diagnosing MLPF noise at initial θ...\n")
    #diag_init = diagnose_mlpf_noise(U_aug_train, U_meas_train, y_training, n_x,
    #                            N_pf, sample_x_0, theta_init;
    #                            n_trials=50, ℓ0=4, ρ_tail=0.45, ℓ_max=9,
    #                            resample_ess_frac=0.7, n_rep_mlpf=3)

    # Optional: also diagnose at true θ (reparam) to see best-case noise
    #@printf("started diagnosing MLPF noise at true θ...\n")
    #diag_true = diagnose_mlpf_noise(U_aug_train, U_meas_train, y_training, n_x,
    #                            N_pf, sample_x_0, θ_true;
    #                            n_trials=50, ℓ0=4, ρ_tail=0.45, ℓ_max=9,
    #                            resample_ess_frac=0.7, n_rep_mlpf=3)

    #targets = (
    #N_pf_list   = [1200, 1600, 2000, 2400, 3000],
    #ℓ0_list     = [3, 4, 5],
    #nrep_list   = [2, 4, 6, 8]
    #)

    #for N in targets.N_pf_list, l0 in targets.ℓ0_list, nr in targets.nrep_list
    #    d = diagnose_mlpf_noise(U_aug_train, U_meas_train, y_training, n_x,
    #                            N, sample_x_0, theta_init;
    #                            n_trials=30, ℓ0=l0, ρ_tail=0.4, ℓ_max=8,
    #                            resample_ess_frac=0.7, n_rep_mlpf=nr)
    #    @printf("N=%d ℓ0=%d nrep=%d | Var=%.2f fracNeg=%.3f meanS=%.3f\n",
    #            N, l0, nr, d.vlog, d.fracNeg, d.meanS)
    #end
# PMMH samples with delta t particle filter
sampling_start = time()

PMMH_samples, acceptance_ratio, all_signs =
    staged_PMMH_dt(U_aug_train, U_meas_train, y_training, n_x, K, K_b, k_d, N_pf,
                   g_theta, sample_x_0, log_pdf_w_theta, logpdf_theta,
                   theta_init, proposal_cov_init, T_chunk, K_stage, alpha;
                   Q=Q, print_progress=true, ℓ0=3, ρ_tail=0.45, ℓ_max=8, 
                   resample_ess_frac=0.7,n_rep_mlpf=2, # cost variance variable
                   variant=VARIANT, h_coarse=H_COARSE, h_fine=H_FINE)
                                                                                # Later increase K 2000 and K_b to 400
elapsed_sampling = time() - sampling_start

Σ_emp = empirical_cov_theta(PMMH_samples; burn=K_b)
    #"Return a Julia-literal string like [a b; c d] (copy-pasteable)."
    function mat_literal(A; digits::Int=16)
        io = IOBuffer()
        print(io, "[")
        for i in 1:size(A,1)
            for j in 1:size(A,2)
                @printf(io, "%.*g", digits, A[i,j])
                if j < size(A,2)
                    print(io, " ")
                end
            end
            if i < size(A,1)
                print(io, "; ")
            end
        end
        print(io, "]")
        return String(take!(io))
    end
    function theta_typical(PMMH_samples; burn::Int=0, method::Symbol=:median)
        K  = length(PMMH_samples)
        i0 = max(1, burn+1)
        Θ  = hcat(getfield.(PMMH_samples, :theta)...)[:, i0:K]  # d × N
        d  = size(Θ, 1)

        θ_typ = zeros(Float64, d)
        if method === :median
            for j in 1:d
                θ_typ[j] = median(view(Θ, j, :))
            end
        elseif method === :mean
            for j in 1:d
                θ_typ[j] = mean(view(Θ, j, :))
            end
        else
            error("method must be :median or :mean")
        end
        return θ_typ
    end
    #"Print copy-pasteable starting config for later runs (coarse/fine/unbiased)."
    function print_starting_points(; theta_init, theta_typ, Σ_emp, regularizer, alpha,
                               N_pf, ℓ0, ρ_tail, ℓ_max, resample_ess_frac, n_rep_mlpf)
        d = length(theta_init)
        Σ_reg = Matrix(Σ_emp) .+ regularizer .* Matrix(I, d, d)
        println("\n# ===== COPY/PASTE STARTING POINTS =====")
        println("theta_typ  = $(repr(theta_typ))   # posterior-typical (post-burn)")
        println("theta_init = $(repr(theta_init))")
        println("d = $(d)")
        println("regularizer = $(regularizer)")
        println("alpha = $(alpha)")
        println("Σ_emp = $(mat_literal(Matrix(Σ_emp)))")
        println("proposal_cov_init = Symmetric($(mat_literal(Σ_reg)))")
        println("")
        println("# PF / MLPF settings used when this was tuned:")
        println("N_pf = $(N_pf)")
        println("ℓ0 = $(ℓ0)")
        println("ρ_tail = $(ρ_tail)")
        println("ℓ_max = $(ℓ_max)")
        println("resample_ess_frac = $(resample_ess_frac)")
        println("n_rep_mlpf = $(n_rep_mlpf)")
        println("# =====================================\n")
    end
    # Example call (fill with your actual current values):
    θ_typ = theta_typical(PMMH_samples; burn=K_b, method=:median)
    print_starting_points(
        theta_init = PMMH_samples[end].theta,
        theta_typ  = θ_typ,
        Σ_emp = Σ_emp,
        regularizer = regularizer,
        alpha = alpha,
        N_pf = N_pf,
        ℓ0 = 3,
        ρ_tail = 0.45,
        ℓ_max = 7,
        resample_ess_frac = 0.7,
        n_rep_mlpf = 2
    )
    diag_post = diagnose_mlpf_noise(U_aug_train, U_meas_train, y_training, n_x,
                                N_pf, sample_x_0, θ_typ;
                                n_trials=30, ℓ0=3, ρ_tail=0.45, ℓ_max=7,
                                resample_ess_frac=0.7, n_rep_mlpf=2)








#=####################################
8) Plots
#################################### =#

# Continuous truth overlay: slice dense arrays to the plotting window
t0_plot = t_train_sorted[end]
t1_plot = t_test_sorted[end]
mask = (t_dense .>= t0_plot - 1e-12) .& (t_dense .<= t1_plot + 1e-12)

t_cont = t_dense[mask]
Y_cont = Y_dense[:, mask]
signs_final = extract_final_signs(all_signs)

#MCMC diagnostics: Autocorrelation and trace plots check mixing
L = length(PMMH_samples)
lag_acf = max(1, min(100, L-1))  # must be < L
lag_ess = max(1, min(200, L-1))  # ditto
plot_autocorrelation(PMMH_samples; max_lag=50, burn=K_b, state_summary=:mean)
Θ = hcat(getfield.(PMMH_samples, :theta)...)  # nθ × L
ScenarioPMCMC.plot_parameter_trace_signed_simplified(PMMH_samples; signs=signs_final, burn=K_b, show_running_signed_mean=true)
#plot_parameter_trace(PMMH_samples; signs=signs_final, burn=K_b)


#Posterior histograms with priors + true values
prior_pdf = Vector{Tuple{Vector{Float64},Vector{Float64}}}()
for i in 1:length(theta_init)
    d = Truncated(Normal(μθ[i], σθ[i]), θ_lo[i], θ_hi[i])
    xs = collect(range(quantile(d, 0.01), stop=quantile(d, 0.99), length=400))
    push!(prior_pdf, (xs, pdf.(d, xs)))
end
for i in 1:length(μx)
    d = Truncated(Normal(μx[i], σx[i]), lo_x[i], hi_x[i])
    xs = collect(range(quantile(d, 0.01), stop=quantile(d, 0.99), length=400))
    push!(prior_pdf, (xs, pdf.(d, xs)))
end
true_vals = [θ_true; X_train[:, end]]
prior_pdf_theta = prior_pdf[1:n_θ]  # only θ prior curves; x(split) has no simple "prior" curve

ScenarioPMCMC.plot_parameter_pdf_split_instrumental(PMMH_samples;
    bins=50, prior_pdf=prior_pdf_theta, true_values=true_vals, burn=K_b)

ScenarioPMCMC.plot_parameter_pdf_split_signed_only(PMMH_samples;
    bins=50, prior_pdf=prior_pdf_theta, true_values=true_vals, signs=signs_final, burn=K_b)


#plot_parameter_pdf(
#    PMMH_samples;
#    bins=50,
#    prior_pdf=prior_pdf,
#    true_values=true_vals,
#    signs=signs_final,
#    burn=K_b,
#    show_signed_density=true,   # recommended for thesis plots
#    clamp_nonneg=false
#)

style_for_display(style::Symbol) = (style === :mlpf_signed ? :em_fine : style)
# ── Add-on: deterministic EM rollout with θ_true under the current VARIANT ──
function em_dense_true(θ::AbstractVector; t0::Real, t1::Real,
                       x0::AbstractVector, style::Symbol=VARIANT)
    # For plots: ALWAYS use fine step
    h_here = H_FINE

    t = collect(range(t0, stop=t1, step=h_here))
    Nd = length(t)

    X = copy(x0)
    Y = Matrix{Float64}(undef, n_y, Nd)
    Y[:, 1] = model.output(X, u_of_t(t[1]), (theta=θ,), 0.0)

    for k in 2:Nd
        Δ  = max(0.0, t[k] - t[k-1])
        uL = u_of_t(t[k-1])  # ZOH on (t[k-1], t[k]]
        X1 = reshape(copy(X), :, 1)

        # deterministic EM: ALWAYS fine h_target for plots
        em_propagate_gap!(X1, uL, θ, Δ; h_target=h_here, add_noise=false)

        X = vec(X1)
        Y[:, k] = model.output(X, u_of_t(t[k]), (theta=θ,), 0.0)
    end
    return t, Y
end

# Roll out from the train/test split with θ_true using the selected VARIANT
t0 = t_train_sorted[end];  t1 = t_test_sorted[end]
t_em, Y_em = em_dense_true(θ_true; t0=t0, t1=t1, x0=X_train[:, end], style=VARIANT)

# Plot: truth (scatter + continuous) vs deterministic EM(θ_true, VARIANT)
for j in 1:n_y
    pEM = plot(
        title  = "Deterministic EM with θ_true — $(String(style_for_display(VARIANT))) — y$(j)",
        xlabel = "t",
        ylabel = "y$(j)",
        legend = :topleft,
        legendfontsize = THESIS_LEGEND_FSIZE,
        guidefontsize  = THESIS_LABEL_FSIZE,
        tickfontsize   = THESIS_TICK_FSIZE,
        titlefontsize  = THESIS_TITLE_FSIZE
    )

    if !isempty(t_train_sorted)
        scatter!(pEM, t_train_sorted, vec(Y_train_true[j, :]);
         ms=THESIS_MS, alpha=0.9, label="true y$(j) (train, clean)")
    end
    if !isempty(t_test_sorted)
        scatter!(pEM, t_test_sorted, vec(Y_test_true[j, :]);
         ms=THESIS_MS, alpha=0.9, label="true y$(j) (test, clean)")
    end

    if !isempty(t_cont)
        plot!(pEM, t_cont, vec(Y_cont[j, :]);
                lw    = THESIS_LW,
                lc    = :black,
                label = "true y$(j) (cont.)")
    end

    plot!(pEM, t_em, vec(Y_em[j, :]);
            lw    = THESIS_LW,
            lc    = :purple,
            label = "EM θ_true ($(String(VARIANT)))")

    vline!(pEM, [0.0]; lc=:black, ls=:dot, lw=THESIS_LW, label="split")


    display(pEM)
end
# --- Tiny utilities ---
const Z095 = 1.6448536269514722  # z-score for 95th percentile (two-sided 90% interval)
trapz(t, f) = sum( (t[2:end] .- t[1:end-1]) .* (f[2:end] .+ f[1:end-1]) )/2
function linear_resample(t_src::Vector, Y_src::AbstractMatrix, t_dst::Vector)
    n_y, Ns = size(Y_src); Nd = length(t_dst)
    Y = Matrix{Float64}(undef, n_y, Nd)
    for j in 1:n_y
        k = 1
        for i in 1:Nd
            while k < Ns && t_src[k+1] <= t_dst[i]; k += 1; end
            if k == Ns
                Y[j, i] = Y_src[j, end]
            else
                α = (t_dst[i]-t_src[k]) / max(t_src[k+1]-t_src[k], eps())
                Y[j, i] = (1-α)*Y_src[j, k] + α*Y_src[j, k+1]
            end
        end
    end
    return Y
end
function best_phase_lag(t::Vector, y_ref::Vector, y_hat::Vector; frac_window=0.25)
    dt = median(diff(t))
    y1 = y_ref .- mean(y_ref)
    y2 = y_hat .- mean(y_hat)
    L  = length(t)
    maxlag = max(1, Int(floor(frac_window * L)))

    bestlag, bestcorr = 0, -Inf
    for lag in -maxlag:maxlag
        if lag ≥ 0
            a = view(y1, 1+lag:L)
            b = view(y2, 1:L-lag)
        else
            a = view(y1, 1:L+lag)
            b = view(y2, 1-lag:L)
        end
        den = sqrt(dot(a,a) * dot(b,b)) + eps()
        c   = dot(a,b) / den
        if c > bestcorr
            bestcorr, bestlag = c, lag
        end
    end

    return bestlag * dt, bestcorr
end





##########################
# Posterior predictive (universal; used by ALL plots)
##########################
# =========================
# Signed posterior predictive summaries
# =========================
const Z80 = 1.2815515655446004  # Gaussian z for central 80% band (10–90)

"""
Compute signed mean and a moment-based band (mean ± Z80*sd) across draw axis (3rd dim).

Y: (n_y × T × Ndraws)
w: length Ndraws, may contain negative values (signs)

Returns: (μ, lo, hi, denom)
If denom ~ 0, falls back to unsigned mean + empirical quantiles (warning).
"""
function signed_mean_band_from_draws(Y::Array{Float64,3}, w::Vector{Float64};
                                     z::Float64 = Z80)
    n_y, T, N = size(Y)
    @assert length(w) == N
    denom = sum(w)

    if abs(denom) < 1e-12
        @printf("[PP signed] WARNING: denom=sum(weights)≈0 -> falling back to unsigned summaries.\n")
        μ  = mean3(Y)
        lo = qfun(Y, 0.10)
        hi = qfun(Y, 0.90)
        return μ, lo, hi, denom
    end

    num  = zeros(Float64, n_y, T)
    num2 = zeros(Float64, n_y, T)

    @inbounds for i in 1:N
        wi = w[i]
        @views begin
            Yi = Y[:, :, i]
            num  .+= wi .* Yi
            num2 .+= wi .* (Yi .^ 2)
        end
    end

    μ  = num ./ denom
    m2 = num2 ./ denom
    v  = max.(m2 .- μ.^2, 0.0)     # clamp negative variance from signed noise
    sd = sqrt.(v)

    lo = μ .- z .* sd
    hi = μ .+ z .* sd
    return μ, lo, hi, denom
end

pretty_time(dt::Real) = @sprintf("%02d:%02d:%05.2f",
                                Int(floor(dt/3600)),
                                Int(floor(mod(dt,3600)/60)),
                                mod(dt,60.0))

@printf("\n⏱️ PMMH finished in %s (%.2f s). Avg. stage acceptance: %.2f%%\n",
        pretty_time(elapsed_sampling), elapsed_sampling, mean(acceptance_ratio))

# one step x_{k} -> x_{k+1} using EM (with or without process noise)
function pp_next_x(θ::AbstractVector, x::AbstractVector, u_aug_col::AbstractVector;
                    h_target::Real,
                    include_process_noise::Bool)
    n_u_phys = model.n_u
    Δt = u_aug_col[n_u_phys+1]
    uL = view(u_aug_col, 1:n_u_phys)
    X = reshape(copy(x), :, 1)
    em_propagate_gap!(X, uL, θ, max(Δt, 0.0);
                        h_target=h_target,
                        add_noise=include_process_noise)
    return vec(X)
end
# Decide which style to DISPLAY (unbiased → show fine EM for visuals)
const ACTIVE_STYLE = VARIANT  # keep for reporting if you want
const STYLE_FOR_DISPLAY = :em_fine
const H_ACTIVE = H_FINE

# For plots: ignore style and always return fine step size
h_for_style(::Symbol) = H_FINE
const N_PF_PP_SPLIT = 250  # plotting-only PF size (tune 150–500)

function split_filter_cloud_for_pp(θ::AbstractVector;
                                   style::Symbol = STYLE_FOR_DISPLAY,
                                   N_pf::Int = N_PF_PP_SPLIT,
                                   seed::Int = 12345)
    rng = MersenneTwister(seed)
    h_target = h_for_style(style)   # :em_fine -> H_FINE, :em_coarse -> H_COARSE

    x_pf, w, logZ = particle_filter_em_dt(
        U_aug_train, U_meas_train, Y_train, n_x, N_pf, sample_x_0, θ;
        h_target = h_target, rng = rng
    )

    Xsplit = copy(@view x_pf[:, end, :])   # n_x × N_pf
    wsplit = copy(@view w[end, :])         # length N_pf
    wsplit ./= sum(wsplit)                 # safety normalize

    return Xsplit, wsplit, logZ
end
# maybe add measurement noise to a single y vector
maybe_add_meas_noise(y::AbstractVector) =
    PP_INCLUDE_MEAS_NOISE ? (y .+ vec(rand(MvNormal(zeros(n_y), R)))) : y

# draw a single state column by PF weights
sample_state_col(M, w) = @views M[:, sample(1:length(w), Weights(vec(w)))]
# --- plotting step helper (uses the same variant as training) ---
# For :em_* we use EM with the matching h (H_FINE or H_COARSE).
plot_step(θ::AbstractVector, x::AbstractVector, u_aug_col::AbstractVector;
          style::Symbol = VARIANT) =
    pp_next_x(θ, x, u_aug_col;
              h_target = h_for_style(style),
              include_process_noise = PP_INCLUDE_PROCESS_NOISE)

# Produce posterior predictive arrays for TRAIN/TEST (EM-consistent)
function posterior_predictive_summaries(PMMH_samples,
                                        U_aug_train::AbstractMatrix,
                                        U_aug_test::AbstractMatrix,
                                        U_meas_test::AbstractMatrix;
                                        style::Symbol = VARIANT)

    S = length(PMMH_samples)
    Ttr, Tte = size(U_aug_train, 2), size(U_aug_test, 2)
    Rdraws = (PP_INCLUDE_STATE_UNC || PP_INCLUDE_PROCESS_NOISE) ? PP_R : 1

    Y_pred_train = Array{Float64}(undef, n_y, Ttr, max(S*Rdraws, S))
    Y_pred_test  = Array{Float64}(undef, n_y, Tte, max(S*Rdraws, S))

    # TRAIN window
    for s in 1:S
        θs = PMMH_samples[s].theta
        R_here = (PP_INCLUDE_STATE_UNC || PP_INCLUDE_PROCESS_NOISE) ? PP_R : 1
        for r in 1:R_here
            x = if PP_INCLUDE_STATE_UNC
                sample_state_col(PMMH_samples[s].x_0, PMMH_samples[s].w_0)
            else
                w0 = PMMH_samples[s].w_0
                (PMMH_samples[s].x_0) * (vec(w0) / sum(w0))
            end
            for k in 1:Ttr
                uk = view(U_aug_train, :, k)
                x  = plot_step(θs, x, uk; style=style)     # ← variant-aware
                yk = g_theta(θs, x, @view(U_meas_train[:, k]))
                Y_pred_train[:, k, (s-1)*R_here + r] = maybe_add_meas_noise(yk)
            end
        end
    end

    # TEST window
    if Tte > 0
        for s in 1:S
            θs = PMMH_samples[s].theta
            R_here = (PP_INCLUDE_STATE_UNC || PP_INCLUDE_PROCESS_NOISE) ? PP_R : 1
            for r in 1:R_here
                x = if PP_INCLUDE_STATE_UNC
                    sample_state_col(PMMH_samples[s].x_m1, PMMH_samples[s].w_m1)
                else
                    wT = PMMH_samples[s].w_m1
                    (PMMH_samples[s].x_m1) * (vec(wT) / sum(wT))
                end
                for k in 1:Tte
                    uk = view(U_aug_test, :, k)
                    x  = plot_step(θs, x, uk; style=style) # ← variant-aware
                    yk = g_theta(θs, x, U_meas_test[:, k])
                    Y_pred_test[:, k, (s-1)*R_here + r] = maybe_add_meas_noise(yk)
                end
            end
        end
    end

    # Summaries
    Ytr_mean = mean3(Y_pred_train)
    Ytr_lo   = qfun(Y_pred_train, 0.05)
    Ytr_hi   = qfun(Y_pred_train, 0.95)

    if Tte > 0
        Yte_mean = mean3(Y_pred_test)
        Yte_lo   = qfun(Y_pred_test, 0.10)
        Yte_hi   = qfun(Y_pred_test, 0.90)
    else
        Yte_mean = zeros(n_y, 0); Yte_lo = zeros(n_y, 0); Yte_hi = zeros(n_y, 0)
    end
    return Ytr_mean, Ytr_lo, Ytr_hi, Yte_mean, Yte_lo, Yte_hi
end


# --- Dense posterior predictive on a fine time grid (continuous curve) ---
function posterior_predictive_dense(PMMH_samples;
                                    t0::Real, t1::Real,
                                    dt::Real=0.1,
                                    style::Symbol = VARIANT,
                                    signs::Union{Nothing,AbstractVector{<:Real}} = nothing,
                                    burn::Int = 0)
    @assert t1 > t0
    S_all = length(PMMH_samples)
    i0 = max(1, burn + 1)
    S = S_all - (i0 - 1)
    @assert S >= 1

    if signs !== nothing
        @assert length(signs) == S_all "signs must align with PMMH_samples."
    end

    t_dense = collect(range(t0, stop=t1, step=dt))
    Nd = length(t_dense)

    R_here = (PP_INCLUDE_STATE_UNC || PP_INCLUDE_PROCESS_NOISE) ? PP_R : 1
    Ndraws = S * R_here

    Y_draws = Array{Float64}(undef, n_y, Nd, Ndraws)
    w_draws = (signs === nothing) ? nothing : Vector{Float64}(undef, Ndraws)

    idx = 0
    for s_idx in i0:S_all
        θs = PMMH_samples[s_idx].theta
        Xsplit, wsplit, _ = split_filter_cloud_for_pp(θs; style=style, seed=20_000 + s_idx)

        for r in 1:R_here
            idx += 1
            if w_draws !== nothing
                w_draws[idx] = float(signs[s_idx])   # same sign for all replicates of this θ
            end

            # start at split
            x = if PP_INCLUDE_STATE_UNC
                sample_state_col(Xsplit, wsplit)
            else
                Xsplit * (vec(wsplit) / sum(wsplit))
            end

            Ytmp = Matrix{Float64}(undef, n_y, Nd)
            Ytmp[:, 1] = maybe_add_meas_noise(
                model.output(x, u_of_t(t_dense[1]), (theta=θs,), 0.0))

            tprev = t_dense[1]
            for k in 2:Nd
                Δ  = max(0.0, t_dense[k] - tprev)
                uL = u_of_t(tprev)

                h_here = h_for_style(style)
                X1 = reshape(copy(x), :, 1)
                em_propagate_gap!(X1, uL, θs, Δ;
                                  h_target=h_here,
                                  add_noise=PP_INCLUDE_PROCESS_NOISE)
                x = vec(X1)

                Ytmp[:, k] = maybe_add_meas_noise(
                    model.output(x, u_of_t(t_dense[k]), (theta=θs,), 0.0))
                tprev = t_dense[k]
            end

            Y_draws[:, :, idx] = Ytmp
        end
    end

    @assert idx == Ndraws

    if signs === nothing
        Y_mean = mean3(Y_draws)
        Y_lo   = qfun(Y_draws, 0.10)
        Y_hi   = qfun(Y_draws, 0.90)
        return t_dense, Y_mean, Y_lo, Y_hi
    else
        Y_mean, Y_lo, Y_hi, _ = signed_mean_band_from_draws(Y_draws, w_draws)
        return t_dense, Y_mean, Y_lo, Y_hi
    end
end


p_u = plot(
    title  = "Measured inputs",
    xlabel = "t",
    ylabel = "u",
    legend = :topleft,
    legendfontsize = THESIS_LEGEND_FSIZE,
    guidefontsize  = THESIS_LABEL_FSIZE,
    tickfontsize   = THESIS_TICK_FSIZE,
    titlefontsize  = THESIS_TITLE_FSIZE,
)

# Color rule: blue/red; green only if more than 2 inputs
input_col(i) = (i == 1 ? :blue : (i == 2 ? :red : :green))

tmin, tmax = first(t_meas), last(t_meas)
for i in 1:n_u
    ci = input_col(i)

    # ZOH curve
    plot!(p_u, t -> u_of_t(t)[i], tmin, tmax;
          label    = "u$(i) (ZOH)",
          linetype = :steppost,
          lw       = THESIS_LW,
          lc       = ci)

    # Measured samples (same color family; no magenta)
    scatter!(p_u, t_meas, vec(U_meas[i, :]);
             label = "u$(i) samples",
             ms    = THESIS_MS,
             alpha = 0.9,
             markercolor = :white,
             markerstrokecolor = ci)
end
display(p_u)


t0 = t_train_sorted[end]
t1 = t_test_sorted[end]

# Dense posterior predictive ONCE for the selected style
t_sel, Y_sel_mean, Y_sel_lo, Y_sel_hi =
    posterior_predictive_dense(PMMH_samples; t0=t0, t1=t1, dt=0.1, style=STYLE_FOR_DISPLAY,
                               signs=signs_final, burn=K_b)

# Lightweight: test-time summaries ONLY (skip train entirely)
function posterior_predictive_test_only(PMMH_samples,
                                        U_aug_test::AbstractMatrix,
                                        U_meas_test::AbstractMatrix;
                                        style::Symbol = STYLE_FOR_DISPLAY,
                                        signs::Union{Nothing,AbstractVector{<:Real}} = nothing,
                                        burn::Int = 0)

    Tte = size(U_aug_test, 2)
    if Tte == 0
        return zeros(n_y, 0), zeros(n_y, 0), zeros(n_y, 0)
    end

    S_all = length(PMMH_samples)
    i0 = max(1, burn + 1)
    S = S_all - (i0 - 1)
    @assert S >= 1

    if signs !== nothing
        @assert length(signs) == S_all "signs must align with PMMH_samples."
    end

    R_here = (PP_INCLUDE_STATE_UNC || PP_INCLUDE_PROCESS_NOISE) ? PP_R : 1
    Ndraws = S * R_here

    Y_draws = Array{Float64}(undef, n_y, Tte, Ndraws)
    w_draws = (signs === nothing) ? nothing : Vector{Float64}(undef, Ndraws)

    idx = 0
    for s_idx in i0:S_all
        θs = PMMH_samples[s_idx].theta
        Xsplit, wsplit, _ = split_filter_cloud_for_pp(θs; style=style, seed=10_000 + s_idx)

        for r in 1:R_here
            idx += 1
            if w_draws !== nothing
                w_draws[idx] = float(signs[s_idx])
            end

            x = if PP_INCLUDE_STATE_UNC
                sample_state_col(Xsplit, wsplit)
            else
                Xsplit * (vec(wsplit) / sum(wsplit))
            end

            for k in 1:Tte
                uk = @view U_aug_test[:, k]
                x  = plot_step(θs, x, uk; style=style)
                yk = g_theta(θs, x, U_meas_test[:, k])
                Y_draws[:, k, idx] = maybe_add_meas_noise(yk)
            end
        end
    end

    @assert idx == Ndraws

    if signs === nothing
        Y_mean = mean3(Y_draws)
        Y_lo   = qfun(Y_draws, 0.10)
        Y_hi   = qfun(Y_draws, 0.90)
        return Y_mean, Y_lo, Y_hi
    else
        Y_mean, Y_lo, Y_hi, _ = signed_mean_band_from_draws(Y_draws, w_draws)
        return Y_mean, Y_lo, Y_hi
    end
end


Yte_mean_sel, Yte_lo_sel, Yte_hi_sel =
    posterior_predictive_test_only(PMMH_samples, U_aug_test, U_meas_test; style=STYLE_FOR_DISPLAY,
                                   signs=signs_final, burn=K_b)

# -------------------- Single plot per output --------------------
for j in 1:n_y
    pY = plot(
        title  = "Signed posterior predictive — $(String(VARIANT)) (plots use H_FINE) — y$(j)",
        xlabel = "t",
        ylabel = "y$(j)",
        legend = :topleft,
        legendfontsize = THESIS_LEGEND_FSIZE,
        guidefontsize  = THESIS_LABEL_FSIZE,
        tickfontsize   = THESIS_TICK_FSIZE,
        titlefontsize  = THESIS_TITLE_FSIZE
    )

    # True data at measurement times
    if !isempty(t_train_sorted)
        scatter!(pY, t_train_sorted, vec(Y_train_true[j, :]);
            ms=THESIS_MS, alpha=0.9, label="true y$(j) (train, clean)",
            mc=:red, markerstrokecolor=:red)
    end
    if !isempty(t_test_sorted)
        scatter!(pY, t_test_sorted, vec(Y_test_true[j, :]);
            ms=THESIS_MS, alpha=0.9, label="true y$(j) (test, clean)",
            mc=:red, markerstrokecolor=:red)
    end

    # Optional continuous truth overlay
    if !isempty(t_cont)
        plot!(pY, t_cont, vec(Y_cont[j, :]);
                lw    = THESIS_LW,
                lc    = :red,
                alpha = 0.45,
                label = "true y$(j) (cont.)")
    end

    # 90% band + mean
    μm  = vec(Yte_mean_sel[j, :])
    q10 = vec(Yte_lo_sel[j, :])
    q90 = vec(Yte_hi_sel[j, :])

    # draw the band BETWEEN quantiles (robust even if μ is outside [q10,q90])
    plot!(pY, t_test_sorted, q10;
            fillrange = q90,
            fillalpha = 0.22,
            fillcolor = :blue,
            lc        = :transparent,
            label     = "Signed moment band (±$(round(Z80,digits=2))σ) @ meas ($(String(STYLE_FOR_DISPLAY)))")


    # overlay mean
    plot!(pY, t_test_sorted, μm;
        lw    = THESIS_LW,
        lc    = :blue,
        label = "Signed PP mean @ meas")

    vline!(pY, [0.0]; lc=:black, ls=:dot, lw=THESIS_LW, label="split")

    display(pY)
end



# ---------- METRICS FOR THE ACTIVE VARIANT (prints only) ----------
# tiny helpers (no deps)

# "ESS of signs" (like ESS of importance weights, but weights are ±1)
ess_sign_factor(s::AbstractVector{<:Real}) =
    (sum(float.(s))^2) / max(sum(abs2, float.(s)), eps())

"""
ESS from an autocorrelation sequence acf[1]=ρ(0)=1, acf[2]=ρ(1), ...

Uses Geyer's initial positive sequence (IPS):
τ = 1 + 2 * sum_{m>=1} (ρ(2m-1)+ρ(2m)) with the sum truncated when a pair becomes non-positive.
ESS = N / τ
"""
function ess_from_acf(acf::AbstractVector{<:Real}, N::Int)
    N <= 1 && return float(N)
    L = length(acf) - 1                 # max lag
    L <= 0 && return float(N)

    s = 0.0
    m = 1
    while (2m) <= L
        ρ1 = float(acf[2m + 0])         # ρ(2m-1) since acf[2]=ρ(1)
        ρ2 = float(acf[2m + 1])         # ρ(2m)
        pair = ρ1 + ρ2
        if !(isfinite(pair)) || pair <= 0
            break
        end
        s += pair
        m += 1
    end
    τ = 1 + 2s
    return float(N) / max(τ, 1e-12)
end 
trapint(t, f) = sum(((f[2:end].^2 .+ f[1:end-1].^2)./2) .* (t[2:end] .- t[1:end-1]))
function interp1(ts::AbstractVector, ys::AbstractVector, tq::AbstractVector)
    out = similar(tq)
    for (i,t) in enumerate(tq)
        k = searchsortedlast(ts, t)
        if k <= 0
            out[i] = ys[1]
        elseif k >= length(ts)
            out[i] = ys[end]
        else
            t0, t1 = ts[k], ts[k+1]; y0, y1 = ys[k], ys[k+1]
            w = (t - t0)/(t1 - t0)
            out[i] = (1-w)*y0 + w*y1
        end
    end
    out
end
# --- signed ACF helpers (pair weights = product is the most defensible default) ---
@inline function _signed_mean(x::AbstractVector{<:Real}, s::AbstractVector{<:Real})
    denom = sum(s)
    return sum(s .* x) / denom, denom
end

function _signed_autocov(x::AbstractVector{<:Real}, s::AbstractVector{<:Real}, k::Int; μ::Real)
    n = length(x)
    num = 0.0
    den = 0.0
    @inbounds for t in 1:(n-k)
        wt = float(s[t]) * float(s[t+k])   # pair-weight = product
        num += wt * (float(x[t]) - μ) * (float(x[t+k]) - μ)
        den += wt
    end
    return (abs(den) < 1e-14) ? 0.0 : (num / den)
end

function _signed_acf_0_to_L(x::AbstractVector{<:Real}, s::AbstractVector{<:Real}, L::Int)
    μ, _ = _signed_mean(x, s)
    c0 = _signed_autocov(x, s, 0; μ=μ)
    if !(isfinite(c0) && c0 > 0)
        return vcat(1.0, fill(0.0, L))  # degenerate
    end
    ρ = Vector{Float64}(undef, L)
    @inbounds for k in 1:L
        ρ[k] = _signed_autocov(x, s, k; μ=μ) / c0
    end
    return vcat(1.0, ρ)
end

# =========================
# Stable signed ESS for θ
#   - ACF of θ uses the MCMC chain only (ignore signs)
#   - sign effect enters via multiplier ess_sign_factor(s)/N
# =========================
function compute_ess_theta_signed(PMMH_samples::AbstractVector,
                                  signs::AbstractVector{<:Real};
                                  max_lag::Int=200,
                                  burn::Int=0)

    K = length(PMMH_samples)
    @assert length(signs) == K

    i0 = max(1, burn + 1)
    s = collect(@view signs[i0:K])
    N = length(s)
    @assert N >= 2 "Need at least 2 post-burn samples."

    nθ = length(PMMH_samples[1].theta)
    Θ = hcat(getfield.(PMMH_samples, :theta)...)[:, i0:K]   # nθ × N

    L = min(max_lag, N-1)
    lags = 0:L

    ess_mcmc   = zeros(Float64, nθ)
    ess_signed = zeros(Float64, nθ)

    essS = ess_sign_factor(s)   # ∈ [0,N]
    mult = essS / N             # ∈ [0,1]

    for j in 1:nθ
        x = vec(@view Θ[j, :])
        acf = autocor(x, lags; demean=true)
        ess_mcmc[j]   = ess_from_acf(acf, N)
        ess_signed[j] = ess_mcmc[j] * mult
    end

    info = (; N, burn,
             ess_sign=essS, mult,
             meanS=mean(s), fracNeg=mean(s .< 0), denom=sum(s))
    return ess_mcmc, ess_signed, info
end

function report_metrics_for_current_variant_signed(PMMH_samples, signs_final;
                                                   use_meas_level_nlpd::Bool = true,
                                                   burn::Int = 0,
                                                   lag_ess::Int = 200)
    @assert !isempty(t_sel) "Need t_sel/Y_sel_* computed before calling metrics."

    # reference truth on test window, interpolated to t_sel
    t0, t1 = t_sel[1], t_sel[end]
    sel_ref = (t_cont .>= t0) .& (t_cont .<= t1)
    t_ref = t_cont[sel_ref]
    Y_ref = Y_cont[:, sel_ref]

    z80 = 1.2815515655446004  # Φ^{-1}(0.9) for central 80% band (10–90%)
    println("\n=== Metrics for $(String(VARIANT)) ===")

    for j in 1:n_y
        μd  = vec(Y_sel_mean[j, :])
        lod = vec(Y_sel_lo[j, :])
        hid = vec(Y_sel_hi[j, :])

        yref_dense = interp1(t_ref, vec(Y_ref[j, :]), t_sel)

        σpred = max.(1e-10, (hid .- lod) ./ (2*z80))
        σtot = use_meas_level_nlpd ? σpred : sqrt.(σpred.^2 .+ σy_eps^2)

        rmse_dense = sqrt(mean((μd .- yref_dense).^2))
        ise_dense  = trapint(t_sel, μd .- yref_dense)
        cov_dense  = mean((yref_dense .>= lod) .& (yref_dense .<= hid))
        bw_dense   = mean(hid .- lod)
        nlp_dense  = 0.5 * mean(log.(2π .* σtot.^2) .+
                                ((yref_dense .- μd).^2) ./ (σtot.^2))

        ω = 3.0
        Tperiod = 2π / ω
        total_window = t_sel[end] - t_sel[1]
        frac_window  = clamp(Tperiod / total_window, 0.05, 0.5)
        lag_s, _ = best_phase_lag(t_sel, yref_dense, μd; frac_window = frac_window)

        rmse_meas = isempty(t_test_sorted) ? NaN :
            sqrt(mean((vec(Y_test_true[j, :]) .- vec(Yte_mean_sel[j, :])).^2))
        cov_meas = isempty(t_test_sorted) ? NaN :
            mean((vec(Y_test_true[j, :]) .>= vec(Yte_lo_sel[j, :])) .&
                 (vec(Y_test_true[j, :]) .<= vec(Yte_hi_sel[j, :])))

        @printf("y%d  RMSE(dense)=%.4f  NLPD=%.4f  cov=%.2f  bandW=%.4f  ISE=%.4f  phase_lag≈%.3fs",
                j, rmse_dense, nlp_dense, cov_dense, bw_dense, ise_dense, lag_s)
        if !isempty(t_test_sorted)
            @printf("  |  RMSE@meas=%.4f  cov@meas=%.2f", rmse_meas, cov_meas)
        end
        println()
    end

    # --- Signed chain diagnostics (single consolidated block) ---
    ess_mcmcθ, ess_signedθ, info = compute_ess_theta_signed(PMMH_samples, signs_final; max_lag=lag_ess, burn=burn)
    μθ_signed, denom, infoμ = signed_mean_theta(PMMH_samples, signs_final; burn=burn)

    acc    = try mean(acceptance_ratio) catch; NaN end
    time_s = try elapsed_sampling catch; NaN end

    @printf("\n=== SIGNED chain diagnostics (burn=%d) ===\n", burn)
    @printf("  avg acceptance: %.2f%%   |  wall time: %.1fs\n", acc, time_s)
    @printf("  mean(sign)=%.4f  fracNeg=%.4f  ess_sign=%.1f (of N=%d)  multiplier=%.4f  denom=%.4e\n",
            info.meanS, info.fracNeg, info.ess_sign, info.N, info.mult, info.denom)
    @printf("  θ ESS_mcmc (min/median): %.1f / %.1f\n", minimum(ess_mcmcθ), median(ess_mcmcθ))
    @printf("  θ ESS_signed (min/median): %.1f / %.1f\n", minimum(ess_signedθ), median(ess_signedθ))
    @printf("  Signed posterior mean θ:\n    %s\n", string(μθ_signed))

    if abs(info.denom) < 1e-6 * info.N
        @printf("  WARNING: denom=sum(signs) is small relative to N -> strong sign cancellations; expect high Monte Carlo error.\n")
    end
end
report_metrics_for_current_variant_signed(PMMH_samples, signs_final; use_meas_level_nlpd=true, burn=K_b, lag_ess=lag_ess)