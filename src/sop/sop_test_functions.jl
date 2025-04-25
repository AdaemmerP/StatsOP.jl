
"""
  crit_val_sop(m, n, alpha, chart_choice, approximate::Bool)

Computes the critical value for the SOP test. Also allows the approximation of 
  the critical value. The input parameters are:

- `m::Int64`: The number of rows in the sop-matrix. Note that the data matrix has 
dimensions `M = m + d₁`, where `d₁` denotes the row delay.
- `n::Int64`: The number of columns in the sop-matrix. Note that the data matrix 
has dimensions `N = n + d₂`, where `d₂` denotes the column delay.
- `alpha::Float64`: The significance level.
- `chart_choice::Int64`: The choice of chart. 
- `approximate::Bool`: If `true`, the approximate critical value is computed. 
If `false`, the exact critical value is computed.

# Examples
```julia-repl
# compute approximate critical value for chart 1 
crit_val_sop(10, 10, 0.05, 1, true)
```
"""
function crit_val_sop(
  M, N, alpha, d1::Int, d2::Int; chart_choice, refinement=0, approximate::Bool=false
)

  # sizes
  m = M - d1
  n = N - d2

  # check whether chart_choice is between 1 and 4
  @assert 1 <= chart_choice <= 7 "Wrong number for test statistic."


  # -------------------------------------------------------------------------#
  # ---------------------- No refinement ------------------------------------#
  # -------------------------------------------------------------------------#
  if refinement == 0
    # compute critical value based on approximation
    if approximate
      if chart_choice == 1
        term = sqrt(4 / 15) # 2 / 9 + 1 / 45
        return (quantile(Normal(0, 1), 1 - alpha / 2) * term)

      elseif chart_choice == 2
        term = sqrt(7 / 9) # 2 / 3 + 1 / 9
        return (quantile(Normal(0, 1), 1 - alpha / 2) * term)

      elseif chart_choice == 3
        term = sqrt(4 / 15) # 2 / 9 + 2 / 45
        return (quantile(Normal(0, 1), 1 - alpha / 2) * term)

      elseif chart_choice == 4
        term = sqrt(32 / 45) # 2 / 3 + 2 / 45
        return (quantile(Normal(0, 1), 1 - alpha / 2) * term)

      end

      # no approximation  
    else

      if chart_choice == 1
        term = sqrt(2 / 9 + 1 / 45 * (1 - 1 / (2 * m) - 1 / (2 * n)))
        return (quantile(Normal(0, 1), 1 - alpha / 2) * term)

      elseif chart_choice == 2
        term = sqrt(2 / 3 + 1 / 9 * (1 - 1 / (2 * m) - 1 / (2 * n)))
        return (quantile(Normal(0, 1), 1 - alpha / 2) * term)

      elseif chart_choice == 3
        term = sqrt(2 / 9 + 2 / 45 * (1 - 1 / (2 * m) - 1 / (2 * n)))
        return (quantile(Normal(0, 1), 1 - alpha / 2) * term)

      elseif chart_choice == 4
        term = sqrt(2 / 3 + 2 / 45 * (1 - 1 / (2 * m) - 1 / (2 * n)))
        return (quantile(Normal(0, 1), 1 - alpha / 2) * term)

      end
    end
  end
  # --------------------------------------------------------------------------

  # -------------------------------------------------------------------------#
  # ---------------------- With refinement ----------------------------------#
  # -------------------------------------------------------------------------#

  if refinement in 1:3

    @assert alpha in (0.1, 0.05, 0.01) "alpha must be 0.1, 0.05 or 0.01"

    if chart_choice == 3

      return ifelse(alpha == 0.1, 3.487299 / (m * n), ifelse(alpha == 0.05, 2.265401 / (m * n), 1.740201 / (m * n)))

    elseif chart_choice == 4

      return ifelse(alpha == 0.1, 2.210104 / (m * n), ifelse(alpha == 0.05, 1.566739 / (m * n), 1.279915 / (m * n)))

    elseif chart_choice == 5

      return ifelse(alpha == 0.1, 2.813519 / (m * n), ifelse(alpha == 0.05, 1.999264 / (m * n), 1.637740 / (m * n)))

    elseif chart_choice == 6

      return ifelse(alpha == 0.1, 2.133017 / (m * n), ifelse(alpha == 0.05, 1.497222 / (m * n), 1.216170 / (m * n)))
    end

  end
end


function test_sop(data, alpha, d1::Int, d2::Int; chart_choice, add_noise::Bool=false, approximate::Bool=false)

  # sizes
  M = size(data, 1)
  N = size(data, 2)
  m = M - d1
  n = N - d2

  # compute critical value
  crit_val = crit_val_sop(
    M, N, alpha, d1, d2; chart_choice=chart_choice, approximate=approximate
  )

  # compute test statistic
  test_stat = stat_sop(
    data, d1, d2;
    chart_choice=chart_choice, add_noise=add_noise
  )

  # return test result
  return (test_stat, crit_val, sqrt(m * n) * abs(test_stat) > crit_val)

end