"""
    particle_filter(u::AbstractMatrix{<:AbstractFloat}, y::AbstractMatrix{<:AbstractFloat}, n_x::Int, N::Int, f::Function, g::Function, sample_v::Function, log_pdf_w::Function, sample_x_0::Function)

Run a particle filter to approximate the log-marginal likelihood ``\\log p(y_{0:t_0-1} \\mid \\theta, \\{u_{0:t_0-1}\\})``.

# Arguments
- `u`: training input trajectory
- `y`: training output trajectory
- `n_x`: number of states
- `N`: number of particles
- `f`: state transition function; has inputs (x, u)
- `g`: measurement function; has inputs (x, u)
- `sample_v`: function that returns N samples from the process noise distribution; has input (N)
- `pdf_v`: probability density function of the process noise; has input (v)
- `log_pdf_w`: function that returns the logarithm of the probability density function of the measurement noise; has input (w)
- `sample_x_0`: function that returns a sample from the distribution over initial states; has no inputs

# Returns
- `x_pf`: state trajectories of particles
- `w`: normalized weights of particles
- `log_likelihood`: log-marginal likelihood estimate
"""
function particle_filter(u::AbstractMatrix{<:AbstractFloat}, y::AbstractMatrix{<:AbstractFloat}, n_x::Int, N::Int, f::Function, g::Function, sample_v::Function, log_pdf_w::Function, sample_x_0::Function)
    # Initialize and pre-allocate.
    T = size(y, 2)
    w = Array{Float64}(undef, T, N)
    x_pf = Array{Float64}(undef, n_x, T, N)
    a = Array{Int64}(undef, T, N)
    log_w = Array{Float64}(undef, 1, N)
    log_likelihood = 0.0

    # Sample initial states.
    for n in 1:N
        x_pf[:, 1, n] .= sample_x_0()
    end

    # Particle filter.
    for t in 1:T
        if t >= 2
            # Resample particles.
            a[t, :] .= sample(1:N, Weights(w[t-1, :]), N)

            # Propagate resampled particles.
            x_pf[:, t, :] .= f(x_pf[:, t-1, a[t, :]], repeat(u[:, t-1], 1, N)) + sample_v(N)
        end

        # PF weight update based on measurement model (logarithms are used for numerical reasons).
        log_w .= log_pdf_w(y[:, t] .- g(x_pf[:, t, :], repeat(u[:, t], 1, N)))
        max_log_w = maximum(log_w)
        w[[t], :] .= exp.(log_w .- max_log_w)
        sum_w = sum(w[t, :])
        w[t, :] .= w[t, :] ./ sum_w

        # Estimate log-likelihood.
        log_likelihood += log(sum_w) + max_log_w - log(N)
    end
    return x_pf, w, log_likelihood
end