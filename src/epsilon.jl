"""
    epsilon(s::Int, K::Int, β::Float)

Determine the parameter ``\\epsilon``. ``1-\\epsilon`` corresponds to a bound on the probability that the incurred cost exceeds the worst-case cost or that the constraints are violated when the input trajectory ``u_{t:t+H}`` is applied to the unknown system.
``\\epsilon`` is the unique solution over the interval ``(0,1)`` of the polynomial equation in the ``v`` variable:

``\\binom{K}{s}(1-v)^{K-s}-\\frac{\\beta}{K}\\sum_{m=s}^{K-1}\\binom{m}{s}(1-v)^{m-s}=0``.

# Arguments
- `s`: cardinality of the support sub-sample 
- `K`: number of scenarios
- `β`: confidence parameter

## References
- S. Garatti and M. C. Campi, “Risk and complexity in scenario optimization,” Mathematical Programming, vol. 191, no. 1, pp. 243–279, 2022.
"""
function epsilon(s::Int, K::Int, β::AbstractFloat)
    alphaU = beta_inc_inv(K - s + 1, s, β)[2]
    m1 = Array(s:K)'
    aux1 = sum(triu(log.(ones(K - s + 1) * m1), 1), dims=2)
    aux2 = sum(triu(log.(ones(K - s + 1) * (m1 .- s)), 1), dims=2)
    coeffs1 = aux2 - aux1
    m2 = Array(K+1:4*K)'
    aux3 = sum(tril(log.(ones(3 * K) .* m2)), dims=2)
    aux4 = sum(tril(log.(ones(3 * K) .* (m2 .- s))), dims=2)
    coeffs2 = aux3 - aux4
    t1 = 0
    t2 = 1 - alphaU
    poly1 = 1 + β / (2 * K) - β / (2 * K) * sum(exp.(coeffs1 .- (K .- m1') * log(t1))) - β / (6 * K) * sum(exp.(coeffs2 .+ (m2' .- K) * log(t1)))
    poly2 = 1 + β / (2 * K) - β / (2 * K) * sum(exp.(coeffs1 .- (K .- m1') * log(t2))) - β / (6 * K) * sum(exp.(coeffs2 .+ (m2' .- K) * log(t2)))
    if !(poly1 * poly2 > 0)
        while t2 - t1 > 1e-10
            t = (t1 + t2) / 2
            polyt = 1 + β / (2 * K) - β / (2 * K) * sum(exp.(coeffs1 .- (K .- m1') * log(t))) - β / (6 * K) * sum(exp.(coeffs2 .+ (m2' .- K) * log(t)))
            if polyt > 0
                t2 = t
            else
                t1 = t
            end
        end
        ϵ = t1
    end
    return ϵ
end