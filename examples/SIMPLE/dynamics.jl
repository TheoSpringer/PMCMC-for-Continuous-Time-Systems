const eps = 1e-4

"""
    smax(a, b)
Smoothened maximum operator max(a,b).
"""
function smax(a, b)
    return (a + b + sqrt((a - b)^2 + eps)) / 2
end

"""
    smin(a, b)
Smoothened minimum operator min(a,b).
"""
function smin(a, b)
    return -smax(-a, -b)
end

"""
    sclip(a, lb, ub)
Smoothened clipping operator max(lb, min(a,ub)).
"""
function sclip(a, lb, ub)
    return smax(lb, smin(a, ub))
end

"""
    fsolar(tau, I50B, parameters)
Compute fraction of solar radiation intercepted by a crop canopy; see Zhao(2019).

# Arguments
- `tau`: cumulative temperature
- `I50B`: leaf senescence
- `parameters`: SIMPLE parameters

# Returns
- fraction of solar radiation intercepted by a crop canopy
"""
function fsolar(tau, I50B, parameters::SIMPLE_parameters)
    fsolar1 = parameters.Rmax / (1 + exp(-0.01 * (tau - parameters.Ia)))
    fsolar2 = parameters.Rmax / (1 + exp(0.01 * (tau - (parameters.tau_sum - I50B))))
    fsolar = smin(fsolar1, fsolar2)
    return fsolar
end

"""
    ftemp(theta, parameters::SIMPLE_parameters)
Compute impact of temperature on biomass growth rate; see Zhao(2019).

# Arguments
- `theta`: temperature
- `parameters`: SIMPLE parameters

# Returns
- impact of temperature on biomass growth rate
"""
function ftemp(theta, parameters::SIMPLE_parameters)
    ftemp1 = 1.0
    ftemp2 = (theta - parameters.theta_base) / (parameters.theta_opt - parameters.theta_base)
    ftemp3 = 0.0
    ftemp = sclip(ftemp2, ftemp3, ftemp1)
    return ftemp
end

"""
    fheat(theta, parameters::SIMPLE_parameters)

Compute the impact of heat stress on biomass growth rate; see Zhao(2019).

# Arguments
- `theta`: temperature (constant over the day)
- `parameters`: SIMPLE parameters

# Returns
- heat stress factor
"""
function fheat(theta, parameters::SIMPLE_parameters)
    fheat1 = 1.0
    fheat2 = 1 - (theta - parameters.theta_heat) / (parameters.theta_ext - parameters.theta_heat)
    fheat3 = 0.0
    fheat = sclip(fheat2, fheat3, fheat1)
    return fheat
end

"""
    fco2(CO2, parameters::SIMPLE_parameters)
Compute the impact of CO₂ on RUE; see Zhao(2019).

# Arguments
- `CO2`: atmospheric CO₂ concentration
- `parameters`: SIMPLE parameters

# Returns
- impact of CO₂ on RUE
"""
function fco2(CO2, parameters::SIMPLE_parameters)
    fco21 = 1 + parameters.Sco2 * 350
    fco22 = 1 + parameters.Sco2 * (CO2 - 350)
    fco23 = 1.0
    fco2 = sclip(fco22, fco23, fco21)
    return fco2
end

"""
    fwater(D, parameters::SIMPLE_parameters)
Compute the drought stress impact on RUE; see Zhao(2019).

# Arguments
- `D`: relative level of drought (ARID index); see Woli (2012)
- `parameters`: SIMPLE parameters

# Returns
- drought stress impact on RUE
"""
function fwater(D, parameters::SIMPLE_parameters)
    fwater1 = 1.0
    fwater2 = 1 - parameters.Swater * D
    fwater3 = 0.0
    fwater = sclip(fwater2, fwater3, fwater1)
    return fwater
end


"""
    fdrought(D, parameters::SIMPLE_parameters)
Compute the drought stress impact on radiation interception; see Zhao(2019).

# Arguments
- `D`: relative level of drought (ARID index); see Woli (2012)
- `parameters`: SIMPLE parameters

# Returns
- drought stress impact on radiation interception
"""
function fdrought(D, parameters::SIMPLE_parameters)
    fdrought1 = 1.0
    fdrought2 = 1.9 - parameters.Swater * D
    fdrought3 = 0.9
    fdrought = sclip(fdrought2, fdrought3, fdrought1)
    return fdrought
end