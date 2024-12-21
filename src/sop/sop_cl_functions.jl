#========================================================================

Multiple Dispatch for 'cl_sop()':
  1. Theoretical distribution
  2. Bootstraping

========================================================================#
"""
    cl_sop(lam, L0, sop_dgp::ICSP, cl_init, reps=10_000; chart_choice=3, jmin=4, jmax=6, verbose=false, d=1)

Compute the control limit for a given in-control distribution. The input parameters are:
  
- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `L0::Float64`: A scalar value for the desired average run length.
- `sop_dgp::ICSP`: A struct for the in-control spatial DGP.
- `cl_init::Float64`: A scalar value for the initial control limit. This is used to find the control limit.
- `reps::Int`: An integer value for the number of repetitions.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4. The default value is 3.
- `jmin::Int`: An integer value for the minimum value for the control limit.
- `jmax::Int`: An integer value for the maximum value for the control limit.
- `verbose::Bool`: A boolean value whether to print the control limit and the average run length.
- `d::Int` An integer value for the embedding dimension. The default value is 1.


```julia-repl
#-- Example
# Parameters
lam = 0.1
L0 = 370
sop_dgp = ICSP(20, 20, Normal(0, 1))
cl_init = 0.5
reps = 10_000
chart_choice = 2
jmin = 4
jmax = 6
verbose = true
d = 1
cl_sop(lam, L0, sop_dgp, cl_init, reps; chart_choice, jmin, jmax, verbose, d)
```
"""
function cl_sop(lam, L0, sop_dgp::ICSP, cl_init, d1::Int, d2::Int, reps=10_000; chart_choice=3, jmin=4, jmax=6, verbose=false)

  L1 = 0.0
  
  for j in jmin:jmax
    for dh in 1:80
      cl_init = cl_init + (-1)^j * dh / 10^j
      L1 = arl_sop(lam, cl_init, sop_dgp, d1, d2, reps; chart_choice)[1]
      if verbose
        println("cl = ", cl_init, "\t", "ARL = ", L1)
      end
      if (j % 2 == 1 && L1 < L0) || (j % 2 == 0 && L1 > L0)
        break
      end
    end
    cl_init = cl_init
  end

  if L1 < L0
    cl_init = cl_init + 1 / 10^jmax
  end

  return cl_init
end


"""
    compute_p_mat(data::Array{Float64,3})

Compute the matrix of p-hat values for a given 3D array of data. These values are used for bootstrapping. 
"""
function compute_p_mat(data::Array{Float64,3}; d1=1, d2=1)

  # pre-allocate
  m = size(data, 1) - d1
  n = size(data, 2) - d2
  lookup_array_sop = compute_lookup_array()
  p_mat = zeros(size(data, 3), 3)
  p_hat = zeros(1, 3)
  sop = zeros(4)
  freq_sop = zeros(Int, 24)
  win = zeros(Int, 4)

  # pre-allocate indexes to compute sum of frequencies
  s_1 = [1, 3, 8, 11, 14, 17, 22, 24]
  s_2 = [2, 5, 7, 9, 16, 18, 20, 23]
  s_3 = [4, 6, 10, 12, 13, 15, 19, 21]

  # compute p_hat matrix
  for i in axes(data, 3)

    # Compute frequencies of sops
    @views data_tmp = data[:, :, i]
    sop_frequencies!(m, n, d1, d2, lookup_array_sop, data_tmp, sop, win, freq_sop)

    # Compute sum of frequencies for each group
    fill_p_hat!(p_hat, chart_choice, sop_freq, m, n,  s_1, s_2, s_3)

    p_mat[i, :] = p_hat

    # Reset win and freq_sop
    fill!(win, 0)
    fill!(freq_sop, 0)
    fill!(p_hat, 0)
  end

  return p_mat
end

#--- Function to critical run length for SOP based on bootstraping
function cl_sop(lam, L0, p_mat::Matrix{Float64}, cl_init, reps=10_000; chart_choice=3, jmin=4, jmax=6, verbose=false)

  L1 = 0.0
  for j in jmin:jmax
    for dh in 1:80
      cl_init = cl_init + (-1)^j * dh / 10^j
      L1 = arl_sop(lam, cl_init, p_mat, reps; chart_choice=chart_choice)[1]
      if verbose
        println("cl = ", cl_init, "\t", "ARL = ", L1)
      end
      if (j % 2 == 1 && L1 < L0) || (j % 2 == 0 && L1 > L0)
        break
      end
    end
    cl_init = cl_init
  end
  if L1 < L0
    cl_init = cl_init + 1 / 10^jmax
  end
  return cl_init
end


function cl_sop(lam, L0, sp_dgp, cl_init, d1_vec::Vector{Int}, d2_vec::Vector{Int}, reps=10_000; chart_choice=3, jmin=4, jmax=6, verbose=false)

  L1 = 0.0
  for j in jmin:jmax
    for dh in 1:80
      cl_init = cl_init + (-1)^j * dh / 10^j
      L1 = arl_sop(lam, cl_init, sp_dgp, d1_vec, d2_vec, reps; chart_choice=chart_choice)[1]
      if verbose
        println("cl = ", cl_init, "\t", "ARL = ", L1)
      end
      if (j % 2 == 1 && L1 < L0) || (j % 2 == 0 && L1 > L0)
        break
      end
    end
    cl_init = cl_init
  end

  if L1 < L0
    cl_init = cl_init + 1 / 10^jmax
  end
  return cl_init
  
end


