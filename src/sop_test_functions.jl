"""
  crit_val_sop(m, n, alpha, chart_choice, approximate::Bool)

Computes the critical value for the SOP test. Also allows the approximation of the critical value. The input parameters are:

- `m::Int64`: The number of rows in the sop-matrix. Note that the data matrix has dimensions `M = m + 1`.
- `n::Int64`: The number of columns in the sop-matrix. Note that the data matrix has dimensions `N = n + 1`.
- `alpha::Float64`: The significance level.
- `chart_choice::Int64`: The choice of chart. 
- `approximate::Bool`: If `true`, the approximate critical value is computed. If `false`, the exact critical value is computed.

# Examples
```julia-repl
# compute approximate critical value for chart 1 
crit_val_sop(10, 10, 0.05, 1, true)
```
"""
function crit_val_sop(m, n, alpha, chart_choice, approximate::Bool)

    if approximate
      if chart_choice == 1
        quantile(Normal(0, 1), 1 - alpha / 2) * sqrt(2 / 9 + 1 / 45)
      elseif chart_choice == 2
        quantile(Normal(0, 1), 1 - alpha / 2) * sqrt(2 / 3 + 1 / 9)
      elseif chart_choice == 3
        quantile(Normal(0, 1), 1 - alpha / 2) * sqrt(2 / 9 + 2 / 45)
      elseif chart_choice == 4
        quantile(Normal(0, 1), 1 - alpha / 2) * sqrt(2 / 3 + 2 / 45)
      end
  
    else
  
      if chart_choice == 1
        quantile(Normal(0, 1), 1 - alpha / 2) * sqrt(2 / 9 + 1 / 45 * (1 - 1 / (2 * m) - 1 / (2 * n)))
      elseif chart_choice == 2
        quantile(Normal(0, 1), 1 - alpha / 2) * sqrt(2 / 3 + 1 / 9 * (1 - 1 / (2 * m) - 1 / (2 * n)))
      elseif chart_choice == 3
        quantile(Normal(0, 1), 1 - alpha / 2) * sqrt(2 / 9 + 2 / 45 * (1 - 1 / (2 * m) - 1 / (2 * n)))
      elseif chart_choice == 4
        quantile(Normal(0, 1), 1 - alpha / 2) * sqrt(2 / 3 + 2 / 45 * (1 - 1 / (2 * m) - 1 / (2 * n)))
      end
    end
  end
  
"""
  crit_val_sop(m, n, alpha, chart_choice, approximate::Bool)

Computes the exact critical value for the SOP test. The input parameters are:

- `m::Int64`: The number of rows in the sop-matrix. Note that the data matrix has dimensions `M = m + 1`.
- `n::Int64`: The number of columns in the sop-matrix. Note that the data matrix has dimensions `N = n + 1`.
- `alpha::Float64`: The significance level.
- `chart_choice::Int64`: The choice of chart. The options are:

# Examples
```julia-repl
# compute approximate critical value for chart 1 
crit_val_sop(10, 10, 0.05, 1, true)
```
"""
  function crit_val_sop(m, n, alpha, chart_choice)
  
      if chart_choice == 1
        quantile(Normal(0, 1), 1 - alpha / 2) * sqrt(2 / 9 + 1 / 45 * (1 - 1 / (2 * m) - 1 / (2 * n)))
      elseif chart_choice == 2
        quantile(Normal(0, 1), 1 - alpha / 2) * sqrt(2 / 3 + 1 / 9 * (1 - 1 / (2 * m) - 1 / (2 * n)))
      elseif chart_choice == 3
        quantile(Normal(0, 1), 1 - alpha / 2) * sqrt(2 / 9 + 2 / 45 * (1 - 1 / (2 * m) - 1 / (2 * n)))
      elseif chart_choice == 4
        quantile(Normal(0, 1), 1 - alpha / 2) * sqrt(2 / 3 + 2 / 45 * (1 - 1 / (2 * m) - 1 / (2 * n)))
      end
  
  end
  
"""
  crit_val_sacf(M, N, alpha)

Computes the critical value for the SACF of lag 1. The input parameters are:

- `M::Int64`: The number of rows in the data matrix.
- `N::Int64`: The number of columns in the data matrix.
- `alpha::Float64`: The significance level.

# Examples
```julia-repl
# compute critical value
crit_val_sacf(11, 11, 0.05)
```
"""
function crit_val_sacf(M, N, alpha)
    quantile(Normal(0, 1), 1 - alpha / 2) / sqrt(M * N)
end