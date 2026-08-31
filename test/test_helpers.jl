# Shared helpers for the test suite.
#
# The hypothesis tests in StatsOrdinalPatterns are randomised: on a single simulated series the reject
# decision is itself a random variable — under H₀ it is `true` with probability `alpha`.
# Asserting `!reject` on one draw therefore fails for ~5% of random seeds, and Julia does
# not guarantee that a given seed reproduces the same random stream across versions.
#
# Statistical assertions in this suite are consequently written in one of two forms:
#   1. structural invariants that hold for *every* draw
#      (e.g. `reject == (stat > crit)`, `0 <= pval <= 1`), or
#   2. aggregate rejection rates over many replications, compared against generous bounds
#      (see `rejection_rate` and `SIZE_*` below).
# Where a single-draw reject/no-reject assertion is used, the alternative is chosen strong
# enough that the power has been verified to be 1.0 over several hundred seeds.

# Bounds for an aggregate size check at nominal alpha = 0.05 with SIZE_REPS replications.
# The Monte Carlo standard error is sqrt(0.05*0.95/500) ≈ 0.0098, so these bounds sit
# about 4 standard errors away on either side: they essentially never fail by chance, but
# still catch a grossly miscalibrated test (one that never rejects, or rejects far too
# often).
const SIZE_REPS = 500
const SIZE_LOWER = 0.01
const SIZE_UPPER = 0.10

"""
    rejection_rate(f, reps)

Fraction of `reps` replications for which `f(seed)` returns `true`. `f` receives the
replication index and is responsible for building its own data from it.
"""
function rejection_rate(f, reps::Int)
  count = 0
  for s in 1:reps
    count += f(s)
  end
  return count / reps
end

"""
    ar1_series(n, phi, seed)

Gaussian AR(1) series. This process is time-reversible, so it carries no up/down
asymmetry — charts such as `UpDownBalance()` have no power against it.
"""
function ar1_series(n, phi, seed)
  rng = MersenneTwister(seed)
  x = zeros(n)
  for t in 2:n
    x[t] = phi * x[t-1] + randn(rng)
  end
  return x
end

"""
    tar1_series(n, seed)

Threshold AR(1) series: serially dependent *and* asymmetric in the up/down direction,
which is the alternative the up/down-asymmetry charts are designed for.
"""
function tar1_series(n, seed)
  rng = MersenneTwister(seed)
  return accumulate((x, e) -> (x >= 0 ? 0.5x : -0.95x) + e, randn(rng, n), init=0.0)
end

"""
    smoothed_image(size_out, seed)

Spatially dependent image obtained by 3x3 moving-average smoothing of white noise.
"""
function smoothed_image(size_out, seed)
  rng = MersenneTwister(seed)
  raw = randn(rng, size_out + 2, size_out + 2)
  return [sum(raw[i:i+2, j:j+2]) / 9 for i in 1:size_out, j in 1:size_out]
end
