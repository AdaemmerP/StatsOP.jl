
"""
    stat_acf(data, h)

Compute the classical sample autocorrelation of the time series `data` at lag `h`,
using `StatsBase.autocor` (demeaned, normalized by the full-sample variance).

- `data`: the time series.
- `h::Int`: lag.
"""
function stat_acf(data, h::Int)
  return autocor(data, [h])[1]
end

"""
    stat_acf(data, lam, null_dist, acf_version)

Compute the sequence of EWMA-smoothed lag-1 autocorrelation (ACF) chart statistics for
the time series `data`.

- `data`: the time series.
- `lam::Float64`: smoothing parameter of the EWMA statistic.
- `null_dist`: in-control (null) distribution; its mean and variance are used to center
  and scale the statistic.
- `acf_version::Int`: version of the ACF statistic:
  1. EWMA of the centered cross products, scaled by an EWMA of the centered squares.
  2. EWMA of the raw cross products, centered and scaled by EWMA estimates of the mean
     and the raw second moment.
  3. EWMA of the centered cross products, scaled by the in-control variance.

Returns a vector with one chart statistic per time step.
"""
function stat_acf(data, lam, null_dist, acf_version)

  # Pre-calculate process parameters
  μ₀ = mean(null_dist)
  σ₀² = var(null_dist)

  number_of_steps = length(data) - 1
  stats_all = zeros(Float64, number_of_steps)

  # Initialize EWMA quantities based on version
  if acf_version == 1
    cₜ = 0.0
    sₜ = σ₀²
  elseif acf_version == 2
    cₜ = μ₀^2
    sₜ = σ₀² + μ₀^2
    mₜ = μ₀
  elseif acf_version == 3
    cₜ = 0.0
  end

  for i in 1:number_of_steps

    x_prev = data[i]
    x_curr = data[i+1]

    if acf_version == 1
      # Equation (3)
      cₜ = lam * (x_curr - μ₀) * (x_prev - μ₀) + (1.0 - lam) * cₜ
      sₜ = lam * (x_curr - μ₀)^2 + (1.0 - lam) * sₜ
      acf_stat = cₜ / sₜ

    elseif acf_version == 2
      # Equation (4)
      cₜ = lam * x_curr * x_prev + (1.0 - lam) * cₜ
      sₜ = lam * x_curr^2 + (1.0 - lam) * sₜ
      mₜ = lam * x_curr + (1.0 - lam) * mₜ
      acf_stat = (cₜ - mₜ^2) / (sₜ - mₜ^2)

    elseif acf_version == 3
      # Equation (5)
      cₜ = lam * (x_curr - μ₀) * (x_prev - μ₀) + (1.0 - lam) * cₜ
      acf_stat = cₜ / σ₀²

    end

    stats_all[i] = acf_stat
  end

  return stats_all
end
