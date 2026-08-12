# --- Result type ---

"""
    ACFTestResult

Result of the asymptotic test based on the classical autocorrelation function
[`test_acf`](@ref).

Fields:
- `stat::Float64`: value of the test statistic.
- `asymp_crit::Float64`: asymptotic critical value.
- `asymp_pval::Float64`: asymptotic p-value.
- `asymp_reject::Bool`: whether the null hypothesis is rejected at the chosen level.
"""
struct ACFTestResult
  stat::Float64
  asymp_crit::Float64
  asymp_pval::Float64
  asymp_reject::Bool
end

function Base.show(io::IO, r::ACFTestResult)
  println(io, "ACFTestResult")
  println(io, "  Statistic:        ", round(r.stat,       digits=4))
  println(io, "  ─────────────────────────────")
  println(io, "  Asymptotic test")
  println(io, "    Critical value: ", round(r.asymp_crit, digits=4))
  println(io, "    p-value:        ", round(r.asymp_pval, digits=4))
  print(io,   "    Reject H₀:      ", r.asymp_reject)
end

# --- Critical value and asymptotic p-value ---

# Under H₀ (white noise), Bartlett's formula gives Var(ρ̂ₕ) ≈ 1/n for any fixed lag h,
# so sqrt(n) * ρ̂ₕ → N(0,1) — the exact 1-D analogue of the sqrt(MN) * ρ̂(d1,d2) → N(0,1)
# result used for the SACF test (see sacf_test_functions.jl). The critical value is
# therefore the same for every lag h.
"""
    crit_val_acf(n, alpha)

Compute the critical value of the asymptotic test based on the classical
autocorrelation function (ACF); see [`test_acf`](@ref). Under the null hypothesis
(white noise), `sqrt(n) * ρ̂ₕ` is asymptotically standard normal for any fixed lag `h`
(Bartlett's formula), so the critical value does not depend on `h`.

- `n::Int`: length of the time series.
- `alpha`: significance level.
"""
function crit_val_acf(n, alpha)
  return quantile(Normal(0, 1), 1 - alpha / 2) / sqrt(n)
end

function _acf_asymp_pval(test_stat, n)
  return 2.0 * (1.0 - cdf(Normal(), abs(test_stat) * sqrt(n)))
end

# --- User-facing test function ---

"""
    test_acf(data, h; alpha=0.05)

Perform the asymptotic two-sided test for serial dependence at lag `h` based on the
classical autocorrelation function (ACF) and return an [`ACFTestResult`](@ref) with the
test statistic, the asymptotic critical value, the p-value, and the reject decision.

- `data`: the time series.
- `h::Int`: lag.
- `alpha=0.05`: significance level.
"""
function test_acf(data, h::Int; alpha=0.05)
  n         = length(data)
  test_stat = stat_acf(data, h)
  crit_val  = crit_val_acf(n, alpha)
  p_val     = _acf_asymp_pval(test_stat, n)
  return ACFTestResult(test_stat, crit_val, p_val, abs(test_stat) > crit_val)
end
