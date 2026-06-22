# --- Critical value for BP-SACF test ---

# Under H₀: MN/2 * bp_stat → Chisq(K), where K = 2w(w+1).
# Reject when MN/2 * bp_stat > χ²_{K, 1-α}, i.e. bp_stat > 2/(MN) * χ²_{K, 1-α}.
function crit_val_sacf_bp(M, N, w, alpha)
  K = 2 * w * (w + 1)
  return 2 / (M * N) * quantile(Chisq(K), 1 - alpha)
end

# --- Result types ---

struct SACFTestResult
  stat::Float64
  asymp_crit::Float64
  asymp_pval::Float64
  asymp_reject::Bool
end

function Base.show(io::IO, r::SACFTestResult)
  println(io, "SACFTestResult")
  println(io, "  Statistic:        ", round(r.stat,       digits=4))
  println(io, "  ─────────────────────────────")
  println(io, "  Asymptotic test")
  println(io, "    Critical value: ", round(r.asymp_crit, digits=4))
  println(io, "    p-value:        ", round(r.asymp_pval, digits=4))
  print(io,   "    Reject H₀:      ", r.asymp_reject)
end

struct SACFBPTestResult
  stat::Float64
  asymp_crit::Float64
  asymp_pval::Float64
  asymp_reject::Bool
end

function Base.show(io::IO, r::SACFBPTestResult)
  println(io, "SACFBPTestResult")
  println(io, "  Statistic:        ", round(r.stat,       digits=4))
  println(io, "  ─────────────────────────────")
  println(io, "  Asymptotic test")
  println(io, "    Critical value: ", round(r.asymp_crit, digits=4))
  println(io, "    p-value:        ", round(r.asymp_pval, digits=4))
  print(io,   "    Reject H₀:      ", r.asymp_reject)
end

# --- Asymptotic p-value helpers ---

# Under H₀ (spatial white noise), the classical large-sample result gives
# sqrt(MN) * ρ̂(d1,d2) → N(0,1) (spatial analogue of the 1-D ACF result,
# variance ≈ 1/T). The two-sided p-value is therefore
# p = 2 * (1 - Φ(|stat| * sqrt(MN))).
function _sacf_asymp_pval(test_stat, M, N)
  return 2.0 * (1.0 - cdf(Normal(), abs(test_stat) * sqrt(M * N)))
end

# bp_stat = 2 * Σ_{k=1}^{K} ρ̂_k² where K = 2w(w+1).
# The factor 2 accounts for the SACF symmetry ρ̂(h1,h2)=ρ̂(-h1,-h2): stat_sacf_bp
# sums over the canonical half-space and doubles (equivalent to the full sum).
# Under H₀ the K half-space lags are asymptotically independent N(0,1/MN), so
# MN * Σ_k ρ̂_k² = Σ_k (sqrt(MN)*ρ̂_k)² → Chisq(K).
# Since bp_stat = 2 * Σ_k ρ̂_k², it follows that MN/2 * bp_stat → Chisq(K).
function _sacf_bp_asymp_pval(test_stat, M, N, w)
  K = 2 * w * (w + 1)
  return 1.0 - cdf(Chisq(K), M * N * test_stat / 2)
end

# --- User-facing test functions ---

function test_sacf(data::Matrix{<:Real}, d1::Int, d2::Int; alpha=0.05)
  M, N      = size(data)
  test_stat = stat_sacf(data, d1, d2)
  crit_val  = crit_val_sacf(M, N, alpha)
  p_val     = _sacf_asymp_pval(test_stat, M, N)
  return SACFTestResult(test_stat, crit_val, p_val, abs(test_stat) > crit_val)
end

function test_sacf_bp(data::Matrix{<:Real}, w::Int; alpha=0.05)
  M, N      = size(data)
  test_stat = stat_sacf_bp(data, w)
  crit_val  = crit_val_sacf_bp(M, N, w, alpha)
  p_val     = _sacf_bp_asymp_pval(test_stat, M, N, w)
  return SACFBPTestResult(test_stat, crit_val, p_val, test_stat > crit_val)
end
