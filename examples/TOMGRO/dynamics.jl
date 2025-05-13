"""
    radiation2ppfd(r::AbstractFloat)
Convert radiation to photosynthetic photon flux density (PPFD).

# Arguments
- `r`: Solar radiation in MJ/m²/day

# Returns
- `ppfd`: Converted PPFD in µmol/m²/s
"""
function radiation2ppfd(r::Union{AbstractFloat,AbstractVector{<:AbstractFloat}})
    return r * 1e6 * 0.48 * 4.57 / 86400
end

"""
    fN(T::AbstractFloat)
Compute scaling fN, which modifies the node development rate depending on the temperature; see Heuvelink(1994) & Jones(1991).

# Arguments
- `T`: Temperature

# Returns
- scaling fN
"""
function fN(T::AbstractFloat)
    if 12 < T <= 28
        return 1.0 + 0.0281 * (T - 28)
    elseif 28 < T < 50
        return 1.0 - 0.0455 * (T - 28)
    else
        return 0.0
    end
end

"""
    dNdt(fN_::AbstractFloat, parameters::TOMGRO_parameters)
Compute node development rate; see Jones(1999).

# Arguments
- `fN_`: modified node development rate
- `parameters`: TOMGRO parameters

# Returns
- node development rate
"""
function dNdt(fN_::AbstractFloat, parameters::TOMGRO_parameters)
    return parameters.Nm * fN_
end

"""
    lambda(Td::AbstractFloat)
Compute scaling lambda, which reduces the rate of leaf area expansion depending on the temperature.

# Arguments
- `Td`: Average daily temperature

# Returns
- scaling lambda
"""
function lambda(Td::AbstractFloat)
    return 1.0
end

"""
    dLAIdt(LAI::AbstractFloat, dens::AbstractFloat, N::AbstractFloat, lambda_::AbstractFloat, dNdt_::AbstractFloat, parameters::TOMGRO_parameters)
Compute derivative of LAI; see Jones(1999).

# Arguments
- `LAI`: Leaf area index
- `dens`: Plant density
- `N`: Number of nodes on mainstem
- `lambda_`: Temperature dependent scaling
- `dNdt_`: Node development rate
- `parameters`: TOMGRO parameters

# Returns
- derivative of LAI
"""
function dLAIdt(LAI::AbstractFloat, dens::AbstractFloat, N::AbstractFloat, lambda_::AbstractFloat, dNdt_::AbstractFloat, parameters::TOMGRO_parameters)
    if LAI > parameters.LAImax
        return 0.0
    else
        a = exp(parameters.beta * (N - parameters.Nb))
        return dens * parameters.sigma * lambda_ * a * dNdt_ / (1 + a)
    end
end

"""
    dWdt(LAI::AbstractFloat, dWfdt_::AbstractFloat, GRnet_::AbstractFloat, dens::AbstractFloat, dNdt_::AbstractFloat, parameters::TOMGRO_parameters)
Compute total dry weight growth rate; see Jones(1999).

# Arguments
- `LAI`: Leaf area index
- `dWfdt_`: growth rate of fruit dry weight
- `GRnet_`: Net aboveground growth rate
- `dens`: Plant density
- `dNdt_`: Node development rate
- `parameters`: TOMGRO parameters

# Returns
- above-ground dry weight growth rate
"""
function dWdt(LAI::AbstractFloat, dWfdt_::AbstractFloat, GRnet_::AbstractFloat, dens::AbstractFloat, dNdt_::AbstractFloat, parameters::TOMGRO_parameters)
    if LAI >= parameters.LAImax
        p1 = 2.0 # Jones(1999)
    else
        p1 = 0.0
    end
    return min(dWfdt_ + (parameters.Vmax - p1) * dens * dNdt_, GRnet_ - p1 * dens * dNdt_)
end

"""
    Df(T::AbstractFloat)
Compute fruit development rate, depending on temperature; see Jones(1991).

# Arguments
- `T`: Temperature

# Returns
- fruit development rate
"""
function Df(T::AbstractFloat)
    if 9 < T <= 28
        return 0.0017 * T - 0.015
    elseif 28 < T <= 35
        return 0.032
    else
        return 0.0
    end
end

"""
    dWmdt(Df_::AbstractFloat, Wf::AbstractFloat, Wm::AbstractFloat, N::AbstractFloat)
Compute the mature fruit growth rate; see Jones(1999).

# Arguments
- `Df_`: Fruit development rate
- `Wf`: Total fruit dry weight
- `Wm`: Mature fruit dry weight
- `N`: Number of nodes on mainstem

# Returns
- mature fruit growth rate
"""
function dWmdt(Df_::AbstractFloat, Wf::AbstractFloat, Wm::AbstractFloat, N::AbstractFloat)
    NFF = 22.0 # Jones(1999)
    kF = 5.0 # Jones(1999)
    if N <= NFF + kF
        return 0.0
    else
        return Df_ * (Wf - Wm)
    end
end

"""
    fR(N::AbstractFloat)
Compute fraction partitioning of biomass to roots; see Jones(1991).

# Arguments
- `N`: Number of nodes on mainstem

# Returns
- root fraction value
"""
function fR(N::AbstractFloat)
    if N >= 30.0
        return 0.07
    else
        return -0.0046 * N + 0.2034
    end
end

"""
    LFmax(CO2::AbstractFloat, parameters::TOMGRO_parameters)
Compute the maximum leaf photosynthetic rate; see Jones(1991).

# Arguments
- `CO2`: CO₂ concentration
- `parameters`: TOMGRO parameters

# Returns
- maximum leaf photosynthetic rate
"""
function LFmax(CO2::AbstractFloat, parameters::TOMGRO_parameters)
    return parameters.tau * CO2
end

"""
    PGRED(T::AbstractFloat)
Compute photosynthetic rate reduction factor under suboptimal temperatures.

# Arguments
- `T`: Temperature

# Returns
- temperature-based reduction factor
"""
function PGRED(T::AbstractFloat)
    if 0 < T <= 12
        return T / 12.0
    elseif 12 < T < 35
        return 1.0
    else
        return 0.0
    end
end

"""
    Pg(LFmax_::AbstractFloat, PGRED_::AbstractFloat, PPFD::AbstractFloat, LAI::AbstractFloat, parameters::TOMGRO_parameters)
Compute photosynthesis rate; see Jones(1991).

# Arguments
- `LFmax_`: Maximum leaf photosynthesis rate
- `PGRED_`: Temperature adjustment factor
- `PPFD`: Photosynthetic photon flux density
- `LAI`: Leaf area index
- `parameters`: TOMGRO parameters

# Returns
- photosynthesis rate
"""
function Pg(LFmax_::AbstractFloat, PGRED_::AbstractFloat, PPFD::AbstractFloat, LAI::AbstractFloat, parameters::TOMGRO_parameters)
    D = 2.593 # coefficient to convert Pg from CO2 to CH2O
    m = 0.1 # leaf light transmission coefficient
    a = D * LFmax_ * PGRED_ / parameters.K
    b = log(((1 - m) * LFmax_ + parameters.Qe * parameters.K * PPFD) /
            ((1 - m) * LFmax_ + parameters.Qe * parameters.K * PPFD * exp(-parameters.K * LAI)))
    return a * b
end

"""
    Rm(T::AbstractFloat, W::AbstractFloat, Wm::AbstractFloat)
Compute maintenance respiration rate; see Jones(1999).

# Arguments
- `T`: Hourly temperature
- `W`: Above-ground dry weight
- `Wm`: Mature fruit dry weight

# Returns
- maintenance respiration rate
"""
function Rm(T::AbstractFloat, W::AbstractFloat, Wm::AbstractFloat)
    Q10 = 1.4 # Jones(1991)
    rm = 0.016 # Jones(1991)
    return Q10^((T - 20) / 10) * rm * (W - Wm)
end

"""
    GRnet(Pg_::AbstractFloat, Rm_::AbstractFloat, fR_::AbstractFloat)
Compute net above-ground growth rate.

# Arguments
- `Pg_`: Photosynthesis rate
- `Rm_`: Maintenance respiration rate
- `fR_`: Root fraction

# Returns
- net growth rate
"""
function GRnet(Pg_::AbstractFloat, Rm_::AbstractFloat, fR_::AbstractFloat)
    E = 0.717 # Dimokas(2009)
    return max(0, E * (Pg_ - Rm_) * (1 - fR_))
end

"""
    fF(Td::AbstractFloat)
Compute fruit partitioning factor; Jones(1991).

# Arguments
- `Td`: Average daily temperature

# Returns
- fruit partitioning factor
"""
function fF(Td::AbstractFloat)
    if 8 < Td <= 28
        return 0.0017 * Td - 0.0147
    elseif Td > 28
        return 0.032
    else
        return 0.0
    end
end

"""
    g(T_daytime::AbstractFloat, parameters::TOMGRO_parameters)
Compute growth reduction factor due to high daytime temperature; see Jones(1999).

# Arguments
- `T_daytime`: Average temperature during daytime hours
- `parameters`: TOMGRO parameters

# Returns
- growth reduction factor
"""
function g(T_daytime::AbstractFloat, parameters::TOMGRO_parameters)
    if T_daytime < parameters.T_CRIT
        return 0.0
    else
        return 1.0 - 0.154 * (T_daytime - parameters.T_CRIT)
    end
end

"""
    dWfdt(GRnet_::AbstractFloat, fF_::AbstractFloat, N::AbstractFloat, g_::AbstractFloat, parameters::TOMGRO_parameters)
Compute the growth rate of fruit dry weight; see Jones(1999).

# Arguments
- `GRnet_`: Net aboveground growth rate
- `fF_`: Fruit partitioning factor
- `N`: Number of nodes on mainstem
- `g_`: Fruit abortion factor
- `parameters`: TOMGRO parameters

# Returns
- fruit dry weight growth rate
"""
function dWfdt(GRnet_::AbstractFloat, fF_::AbstractFloat, N::AbstractFloat, g_::AbstractFloat, parameters::TOMGRO_parameters)
    NFF = 22.0 # nodes per plant when first fruit appears
    # fF_ = 0.5
    if N <= NFF
        return 0.0
    end
    return GRnet_ * parameters.alpha_F * fF_ * (1 - exp(parameters.v * (NFF - N))) * g_
end