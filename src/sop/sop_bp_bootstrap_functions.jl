export bootstrap_sop_bp

# ------------------------------------------------------------------------------
# 1. STATISTIC CALCULATION
# ------------------------------------------------------------------------------
# SOPBuffer and resample_2d! are defined in sop_bootstrap_functions.jl (included earlier)
function stat_sop_bp!(buffer::SOPBuffer, data, w::Int; chart_choice, refinement=false)
  (; lookup_array_sop, sop, win, sop_freq, p_hat, index_sop) = buffer

  M_rows = size(data, 1)
  N_cols = size(data, 2)
  bp_stat = 0.0

  for (d1, d2) in Iterators.product(1:w, 1:w)
    fill!(sop_freq, 0)
    fill!(p_hat, 0)
    fill!(win, 0)

    m = M_rows - d1
    n = N_cols - d2

    sop_frequencies!(m, n, d1, d2, lookup_array_sop, data, sop, win, sop_freq)
    fill_p_hat!(p_hat, chart_choice, refinement, sop_freq, m, n, index_sop)
    bp_stat += chart_stat_sop(p_hat, chart_choice)^2
  end

  return bp_stat
end

# ------------------------------------------------------------------------------
# 2. BOOTSTRAP WRAPPER
# ------------------------------------------------------------------------------
"""
    bootstrap_sop_bp(
      data::Matrix{<:Real}, n_boot::Int, w::Int;
      chart_choice=TauTilde(), refinement=false, block_size::Int=1
    )

Generate a bootstrap distribution of the BP-SOP statistic for a single spatial image.

- `data`: The 2D image (M × N matrix).
- `n_boot`: Number of bootstrap replications.
- `w::Int`: Window size; all (d1, d2) ∈ {1,…,w}² are included in the BP sum.
- `chart_choice`: The chart statistic type (e.g. `TauTilde()`, `KappaTilde()`).
- `refinement`: `false` for classical types, or a `RefinedType` instance.
- `block_size`: Set > 1 to use 2D block bootstrap and preserve spatial dependencies.
"""
function bootstrap_sop_bp(
  data::Matrix{<:Real},
  n_boot::Int,
  w::Int;
  chart_choice=TauTilde(),
  refinement=false,
  block_size::Int=1
)

  M, N = size(data)
  results = zeros(Float64, n_boot)
  buffer = SOPBuffer(M, N; refinement=refinement)

  for b in 1:n_boot
    resample_2d!(buffer.resampled_data, data, block_size)
    results[b] = stat_sop_bp!(buffer, buffer.resampled_data, w;
      chart_choice=chart_choice, refinement=refinement)
  end

  return results
end
