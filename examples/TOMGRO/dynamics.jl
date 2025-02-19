# filepath: /plant-growth-simulation/plant-growth-simulation/src/TOMGRO.jl
using Plots

"""
# Tomato Growth Model
This script simulates the growth of tomato plants based on environmental factors such as temperature, radiation, and CO₂ concentration. 
It models various aspects of plant development, including node formation, leaf area expansion, dry matter accumulation, and fruit development.

Reference:
- CODE: https://gist.github.com/gyosit/abeab4e595d7ddcd65b55c1270d240c8
- Jones (1999) "Reduced state-variable tomato growth model"
- Jones (1991) "A dynamic tomato growth and yield model (TOMGRO)"
- Dimokas (2009) "Calibration and validation of a biological model to simulate the development and production of tomatoes in Mediterranean greenhouses during winter period"
- Heuvelink (1994) "Dry-matter partitioning in a tomato crop: Comparison of two simulation models"
"""

"""
# Parameters
Constants used in the tomato growth model.

- `Nm`: Maximum rate of node appearance (at optimal temperatures)
- `Nb`: Coefficient in expolinear equation, projection of linear segment of LAI vs N to horizontal axis
- `sigma`: Maximum leaf area expansion per node, coefficient in expolinear equation
- `beta`: Coefficient in expolinear equation
- `Vmax`: Maximum increase in vegetative tissue d.w. growth per node
- `Qe`: Leaf quantum efficiency
- `tau`: Carbon dioxide use efficiency
- `K`: Light extinction coefficient
- `CE`: Conversion coefficient for assimilated carbon into dry matter
- `T_CRIT`: Mean daytime temperature above which fruit abortion starts
- `alpha_F`: Maximum partitioning of new growth to fruit
- `v`: Transition coefficient governing the shift between vegetative and reproductive growth phases
- `LAImax`: Maximum leaf area index
"""
const Nm = 0.495
const Nb = 13
const sigma = 0.041
const beta = 0.22
const Vmax = 6
const Qe = 0.09
const tau = 0.12
const K = 0.61
const CE = 0.74
const T_CRIT = 24
const alpha_F = 0.95
const v = 0.24
const LAImax = 6.0

"""
# Initial State
Initial values for model state variables.

- `N`: Number of nodes on mainstem
- `LAI`: Leaf area index
- `W`: Above-ground dry weight
- `Wm`: Mature fruit dry weight
- `Wf`: Total fruit dry weight
"""
const N_init = 10.0
const LAI_init = 0.05
const W_init = 0.0
const Wm_init = 0.0
const Wf_init = 0.0

"""
Convert radiation to photosynthetic photon flux density (PPFD).

# Arguments
- `r`: Solar radiation in MJ/m²/day

# Returns
- `ppfd`: Converted PPFD in µmol/m²/s
"""
function radiation2ppfd(r)
    return r * 1e6 * 0.48 * 4.57 / 86400
end

"""
Compute scaling fN, which modifies the node development rate depending on the temperature; see Heuvelink(1994) & Jones(1991).

# Arguments
- `T`: Temperature

# Returns
- scaling fN
"""
function fN(T)
    if 12 < T <= 28
        return 1.0 + 0.0281 * (T - 28)
    elseif 28 < T < 50
        return 1.0 - 0.0455 * (T - 28)
    else
        return 0.0
    end
end

"""
Compute node development rate; see Jones(1999).

# Arguments
- `fN_`: modified node development rate

# Returns
- node development rate
"""
function dNdt(fN_)
    return Nm * fN_
end

"""
Compute scaling lambda, which reduces the rate of leaf area expansion depending on the temperature.

# Arguments
- `Td`: Average daily temperature

# Returns
- scaling lambda
"""
function lambda(Td)
    return 1.0
end

"""
Compute derivative of LAI; see Jones(1999).

# Arguments
- `LAI`: Leaf area index
- `dens`: Plant density
- `N`: Number of nodes on mainstem
- `lambda_`: Temperature dependent scaling
- `dNdt_`: Node development rate

# Returns
- derivative of LAI
"""
function dLAIdt(LAI, dens, N, lambda_, dNdt_)
    if LAI > LAImax
        return 0.0
    else
        a = exp(beta * (N - Nb))
        return dens * sigma * lambda_ * a * dNdt_ / (1 + a)
    end
end

"""
Compute total dry weight growth rate; see Jones(1999).

# Arguments
- `LAI`: Leaf area index
- `dWfdt_`: growth rate of fruit dry weight
- `GRnet_`: Net aboveground growth rate
- `dens`: Plant density
- `dNdt_`: Node development rate

# Returns
- Above-ground dry weight growth rate
"""
function dWdt(LAI, dWfdt_, GRnet_, dens, dNdt_)
    if LAI >= LAImax
        p1 = 2.0 # Jones(1999)
    else
        p1 = 0.0
    end
    return min(dWfdt_ + (Vmax - p1) * dens * dNdt_, GRnet_ - p1 * dens * dNdt_)
end

"""
Compute fruit development rate, depending on temperature; see Jones(1991).

# Arguments
- `T`: Temperature

# Returns
- Fruit development rate
"""
function Df(T)
    if 9 < T <= 28
        return 0.0017 * T - 0.015
    elseif 28 < T <= 35
        return 0.032
    else
        return 0.0
    end
end

"""
Compute the mature fruit growth rate; see Jones(1999).

# Arguments
- `Df_`: Fruit development rate
- `Wf`: Total fruit dry weight
- `Wm`: Mature fruit dry weight
- `N`: Number of nodes on mainstem

# Returns
- mature fruit growth rate
"""
function dWmdt(Df_, Wf, Wm, N)
    NFF = 22.0 # Jones(1999)
    kF = 5.0 # Jones(1999)
    if N <= NFF + kF
        return 0.0
    else
        return Df_ * (Wf - Wm)
    end
end

"""
Compute fraction partitioning of biomass to roots; see Jones(1991).

# Arguments
- `N`: Number of nodes on mainstem

# Returns
- Root fraction value
"""
function fR(N)
    if N >= 30
        return 0.07
    else
        return -0.0046 * N + 0.2034
    end
end

"""
Compute the maximum leaf photosynthetic rate; see Jones(1991).

# Arguments
- `CO2`: CO₂ concentration

# Returns
- Maximum leaf photosynthetic rate
"""
function LFmax(CO2)
    return tau * CO2
end

"""
Compute photosynthetic rate reduction factor under suboptimal temperatures.

# Arguments
- `T`: Temperature

# Returns
- Temperature-based reduction factor
"""
function PGRED(T)
    if 0 < T <= 12
        return T / 12.0
    elseif 12 < T < 35
        return 1.0
    else
        return 0.0
    end
end

"""
Compute photosynthesis rate; see Jones(1991).

# Arguments
- `LFmax_`: Maximum leaf photosynthesis rate
- `PGRED_`: Temperature adjustment factor
- `PPFD`: Photosynthetic photon flux density
- `LAI`: Leaf area index

# Returns
- Photosynthesis rate
"""
function Pg(LFmax_, PGRED_, PPFD, LAI)
    D = 2.593 # coefficient to convert Pg from CO2 to CH2O
    m = 0.1 # leaf light transmission coefficient
    a = D * LFmax_ * PGRED_ / K
    b = log(((1 - m) * LFmax_ + Qe * K * PPFD) /
            ((1 - m) * LFmax_ + Qe * K * PPFD * exp(-K * LAI)))
    return a * b
end

"""
Compute maintenance respiration rate; see Jones(1999).

# Arguments
- `T`: Hourly temperature
- `W`: Above-ground dry weight
- `Wm`: Mature fruit dry weight

# Returns
- Maintenance respiration rate
"""
function Rm(T, W, Wm)
    Q10 = 1.4 # Jones(1991)
    rm = 0.016 # Jones(1991)
    return Q10^((T - 20) / 10) * rm * (W - Wm)
end

"""
Compute net above-ground growth rate.

# Arguments
- `Pg_`: Photosynthesis rate
- `Rm_`: Maintenance respiration rate
- `fR_`: Root fraction

# Returns
- Net growth rate
"""
function GRnet(Pg_, Rm_, fR_)
    E = 0.717 # Dimokas(2009)
    return max(0, E * (Pg_ - Rm_) * (1 - fR_))
end

"""
Compute fruit partitioning factor; Jones(1991).

# Arguments
- `Td`: Average daily temperature

# Returns
- Fruit partitioning factor
"""
function fF(Td)
    if 8 < Td <= 28
        return 0.0017 * Td - 0.0147
    elseif Td > 28
        return 0.032
    else
        return 0.0
    end
end

"""
Compute growth reduction factor due to high daytime temperature; see Jones(1999).

# Arguments
- `T_daytime`: Average temperature during daytime hours

# Returns
- Growth reduction factor
"""
function g(T_daytime)
    if T_daytime < T_CRIT
        return 0.0
    else
        return 1.0 - 0.154 * (T_daytime - T_CRIT)
    end
end

"""
Simulate the growth process over time.

# Arguments
- `inT`: Temperatures over time
- `inPPFD`: Photosynthetic photon flux density over time
- `inCO2`: CO₂ concentration over time

# Returns
- A dictionary containing growth-related variables over time
"""
function calc(inT, inPPFD, inCO2)
    # Initial values
    N = N_init
    LAI = LAI_init
    W = W_init
    Wm = Wm_init
    Wf = Wf_init

    # Growth data per day
    N_hist = Float64[]
    LAI_hist = Float64[]
    W_hist = Float64[]
    Wm_hist = Float64[]
    Wf_hist = Float64[]

    # Growth rate per day
    delN = Float64[]
    delLAI = Float64[]
    delW = Float64[]
    delWm = Float64[]
    delWf = Float64[]

    # Simulation length in h
    sim_length = Int(floor(length(inT) / 24)) * 24

    for i in 1:24:sim_length
        # Reset variables
        dNdt_ = 0.0
        Td = 0.0
        Tdaytime = 0.0
        PPFDd = 0.0

        # Calculate daily temperature and PPFD
        for h in 1:24
            Td += inT[i+h-1]
            PPFDd += inPPFD[i+h-1]
            if h == 14  # 14:00
                Tdaytime = inT[i+h-1]
            end
        end
        Td /= 24
        PPFDd /= 24

        # dN/dt
        fN_ = fN(Td)
        dNdt_ += dNdt(fN_)

        # d(LAI)/dt
        lambda_ = lambda(Td)
        dLAIdt_ = dLAIdt(LAI, 3.10, N, lambda_, dNdt_)

        # dWfdt
        fR_ = fR(N)
        LFmax_ = LFmax(inCO2[i])
        PGRED_ = PGRED(Td)
        Pg_ = Pg(LFmax_, PGRED_, PPFDd, LAI)
        Rm_ = Rm(Td, W, Wm)
        GRnet_ = GRnet(Pg_, Rm_, fR_)
        fF_ = fF(Td)
        g_ = g(Tdaytime)
        dWfdt_ = dWfdt(GRnet_, fF_, N, g_)

        # dWdt
        dWdt_ = dWdt(LAI, dWfdt_, GRnet_, 3.10, dNdt_)

        # dWmdt
        Df_ = Df(Td)
        dWmdt_ = dWmdt(Df_, Wf, Wm, N)

        # Update variables
        N += dNdt_
        LAI += dLAIdt_
        Wf += dWfdt_
        W += dWdt_
        Wm += dWmdt_

        # Save
        push!(N_hist, N)
        push!(LAI_hist, LAI)
        push!(W_hist, W)
        push!(Wf_hist, Wf)
        push!(Wm_hist, Wm)
        push!(delN, dNdt_)
        push!(delLAI, dLAIdt_)
        push!(delW, dWdt_)
        push!(delWf, dWfdt_)
        push!(delWm, dWmdt_)
    end

    return Dict(
        "N" => N, "LAI" => LAI, "Wf" => Wf, "W" => W, "Wm" => Wm,
        "N_hist" => N_hist, "LAI_hist" => LAI_hist, "W_hist" => W_hist, "Wf_hist" => Wf_hist, "Wm_hist" => Wm_hist,
        "delN" => delN, "delLAI" => delLAI, "delWf" => delWf, "delW" => delW, "delWm" => delWm
    )
end

"""
Generate pseudo climate data for simulation.

# Arguments
- `days`: Length of the simulation in days

# Returns
- A dictionary containing climate conditions over time
"""
function pseudoClimate(days=100)
    T = fill(25.0, days * 24)
    RAD = 35 .* (sin.(range(0, days * 2π, length=days * 24)) ./ 2 .+ 0.5)
    PPFD = radiation2ppfd.(RAD)
    CO2 = fill(400.0, days * 24)
    return Dict("T" => T, "PPFD" => PPFD, "RAD" => RAD, "CO2" => CO2)
end

"""
Plot the input climate data.

# Arguments
- `datas`: Dictionary containing environmental data (`T`, `PPFD`, `RAD`, `CO2`)
"""
function inputPlot(datas)
    T = datas["T"]
    PPFD = datas["PPFD"]
    RAD = datas["RAD"]
    CO2 = datas["CO2"]
    t = range(0, length(T) / 24, length=length(T))

    p1 = plot(t, T, xlabel="day", ylabel="Temperature in °C", title="Temperature Over Time", lw=2)
    display(p1)
    p2 = plot(t, PPFD, xlabel="day", ylabel="PPFD in μmol/m²/s", title="PPFD Over Time", lw=2)
    display(p2)
    p3 = plot(t, RAD, xlabel="day", ylabel="Radiation in MJ/m²/day", title="Radiation Over Time", lw=2)
    display(p3)
    p4 = plot(t, CO2, xlabel="day", ylabel="CO2 in ppm", title="CO2 Concentration Over Time", lw=2)
    display(p4)
end

if abspath(PROGRAM_FILE) == @__FILE__
    period = 100
    datas = pseudoClimate(period)
    results = calc(datas["T"], datas["PPFD"], datas["CO2"])

    for (key, label) in [("N_hist", "Number of nodes on mainstem"),
        ("LAI_hist", "Leaf AREA Index (m²/m²)"),
        ("W_hist", "Above Ground dry weight (g/m²)"),
        ("Wf_hist", "Total fruit dry weight (g/m²)"),
        ("Wm_hist", "Mature fruit dry weight (g/m²)")]
        p = plot(results[key], xlabel="day", title="TOMGRO", ylabel=label, lw=2)
        display(p)
    end
    inputPlot(datas)
end