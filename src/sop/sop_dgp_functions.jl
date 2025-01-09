

# Method to initialize matrix for SAR(1,1) with continuous errors
function init_mat!(dgp::SAR11, dist_error, dgp_params, mat)

  μ = mean(dist_error)
  α₁ = dgp_params[1]
  α₂ = dgp_params[2]
  α₃ = dgp_params[3]

  μₓ = μ / (1 - α₁ - α₂ - α₃)
  mat[1, :] .= μₓ
  mat[:, 1] .= μₓ

end

# Method to initialize matrix for SAR(2,2) with continuous errors
function init_mat!(dgp::SAR22, dist_error, dgp_params, mat)

  μ = mean(dist_error)
  α₁ = dgp_params[1]
  α₂ = dgp_params[2]
  α₃ = dgp_params[3]
  α₄ = dgp_params[4]
  α₅ = dgp_params[5]
  α₆ = dgp_params[6]
  α₇ = dgp_params[7]
  α₈ = dgp_params[8]

  μₓ = μ / (1 - α₁ - α₂ - α₃ - α₄ - α₅ - α₆ - α₇ - α₈)
  mat[1:2, :] .= μₓ
  mat[:, 1:2] .= μₓ

end

# Method to initialize matrix for SAR(1,1) with discrete errors
function init_mat!(dgp::SINAR11, dist_error, dgp_params, mat)

  μ = mean(dist_error)
  α₁ = dgp_params[1]
  α₂ = dgp_params[2]
  α₃ = dgp_params[3]

  μₓ = μ / (1 - α₁ - α₂ - α₃)
  μₓ = Int(round(μₓ)) # round and convert to integer
  mat[1, :] .= μₓ
  mat[:, 1] .= μₓ

end

# Initialize for SQMA(1, 1) process -> do nothing
function init_mat!(dgp::SQMA11, dist_error, dgp_params, mat)
  # Initialization not necessary
end

# Initialize for SQINMA(1, 1) -> do nothing
function init_mat!(dgp::SQINMA11, dist_error, dgp_params, mat)
  # Initialization not necessary
end

# Initialize for SQMA(2, 2) -> do nothing
function init_mat!(dgp::SQMA22, dist_error, dgp_params, mat)
  # Initialization not necessary
end

# Initialize for BSQMA(1, 1) -> do nothing
function init_mat!(dgp::BSQMA11, dist_error, dgp_params, mat)
  # Initialization not necessary
end

# Compute once matrix for SAR(1) process
function build_sar1_matrix(dgp::SAR1)

  margin = dgp.margin
  M = dgp.M_rows + 2 * margin
  N = dgp.N_cols + 2 * margin

  α₁ = dgp.dgp_params[1]
  α₂ = dgp.dgp_params[2]
  α₃ = dgp.dgp_params[3]
  α₄ = dgp.dgp_params[4]

  B = zeros(M * N, M * N)

  for i in 1:M
    for j in 1:N

      index = M * (j - 1) + i

      if i > 1
        B[index, M*(j-1)+i-1] = α₁
      end

      if j > 1
        B[index, M*(j-2)+i] = α₂
      end

      if j < N
        B[index, M*j+i] = α₃
      end

      if i < M
        B[index, M*(j-1)+i+1] = α₄
      end
    end
  end

  I_mat = I(M * N)
  B .= (B .- I_mat)
  B .= inv(B)

  return B

end

# Method to fill data matrix for SAR(1) without additive outliers
function fill_mat_dgp_sop!(
  dgp::SAR1, dist_error, dist_ao::Nothing, mat, mat_ao::Matrix{Float64},
  vec_ar::Vector{Float64}, vec_ar2::Vector{Float64}, mat2::Matrix{Float64}
)

  # draw MA-errors  
  margin = dgp.margin
  M_rows = dgp.M_rows
  N_cols = dgp.N_cols

  # M = M_rows + 2 * margin
  # N = N_cols + 2 * margin
  #m = dgp.m_rows
  #n = dgp.n_cols
  #M = m + 1 + 2 * margin
  #N = n + 1 + 2 * margin

  rand!(dist_error, vec_ar)
  mul!(vec_ar2, mat, vec_ar)

  mat2[:] = vec_ar2 # reshape(vec_ar2, M, N)

  return @views mat2[(margin+1):(margin+M_rows), (margin+1):(margin+N_cols)]

end

# Method to fill matrix for SAR(1) with additive outliers
function fill_mat_dgp_sop!(
  dgp::SAR1, dist_error::UnivariateDistribution, dist_ao::UnivariateDistribution,
  mat, mat_ao::Matrix{Float64}, vec_ar::Vector{Float64}, vec_ar2::Vector{Float64}
)

  # draw MA-errors  
  margin = dgp.margin
  M_rows = dgp.M_rows
  N_cols = dgp.N_cols

  # M = M_rows + 2 * margin
  # N = N_cols + 2 * margin
  # m = dgp.M_rows 
  # n = dgp.N_cols
  # M = m + 1 + 2 * margin
  # N = n + 1 + 2 * margin

  rand!(dist_error, vec_ar)
  mul!(vec_ar2, mat, vec_ar)
  mat2[:] = vec_ar2 # reshape(vec_ar2, M, N)

  # add AO
  rand!(dist_ao, mat_ao)
  mat2 .= mat2 .+ mat_ao

  return @views mat2[(margin+1):(margin+M_rows), (margin+1):(margin+N_cols)]

end

# Method to fill matrix for SAR(1,1) without additive outliers
function fill_mat_dgp_sop!(
  dgp::SAR11, dist_error::ContinuousUnivariateDistribution, dist_ao::Nothing,
  mat, mat_ao::Matrix{Float64}, mat_ma::Matrix{Float64}
)

  # Extract parameters
  prerun = dgp.prerun
  α₁ = dgp.dgp_params[1]
  α₂ = dgp.dgp_params[2]
  α₃ = dgp.dgp_params[3]

  # Fill
  for t2 in 2:size(mat, 2)
    for t1 in 2:size(mat, 1)
      mat[t1, t2] = α₁ * mat[t1-1, t2] + α₂ * mat[t1, t2-1] + α₃ * mat[t1-1, t2-1] + rand(dist_error)
    end
  end

  return @views mat[(prerun+1):end, (prerun+1):end]

end

# Method to fill matrix for SAR(1,1) with additive outliers
function fill_mat_dgp_sop!(
  dgp::SAR11, dist_error::ContinuousUnivariateDistribution, dist_ao::UnivariateDistribution,
  mat, mat_ao::Matrix{Float64}, mat_ma::Matrix{Float64}
)

  # Extract parameters
  prerun = dgp.prerun
  α₁ = dgp.dgp_params[1]
  α₂ = dgp.dgp_params[2]
  α₃ = dgp.dgp_params[3]

  # Fill mat
  for t2 in 2:size(mat, 2)
    for t1 in 2:size(mat, 1)
      mat[t1, t2] = α₁ * mat[t1-1, t2] + α₂ * mat[t1, t2-1] + α₃ * mat[t1-1, t2-1] + rand(dist_error)
    end
  end

  # Add AOs
  rand!(dist_ao, mat_ao)
  mat .= mat .+ mat_ao

  return @views mat[(prerun+1):end, (prerun+1):end]

end

# Method to fill matrix for SAR(2,2) without additive outliers
function fill_mat_dgp_sop!(
  dgp::SAR22, dist_error::ContinuousUnivariateDistribution, dist_ao::Nothing,
  mat, mat_ao::Matrix{Float64}, mat_ma::Matrix{Float64}
)

  # Extract parameters
  prerun = dgp.prerun
  α₁ = dgp.dgp_params[1]
  α₂ = dgp.dgp_params[2]
  α₃ = dgp.dgp_params[3]
  α₄ = dgp.dgp_params[4]
  α₅ = dgp.dgp_params[5]
  α₆ = dgp.dgp_params[6]
  α₇ = dgp.dgp_params[7]
  α₈ = dgp.dgp_params[8]

  # Fill
  for t2 in 3:size(mat, 2)
    for t1 in 3:size(mat, 1)
      mat[t1, t2] = α₁ * mat[t1-1, t2] +
                    α₂ * mat[t1, t2-1] +
                    α₃ * mat[t1-1, t2-1] +
                    α₄ * mat[t1-2, t2] +
                    α₅ * mat[t1, t2-2] +
                    α₆ * mat[t1-2, t2-1] +
                    α₇ * mat[t1-1, t2-2] +
                    α₈ * mat[t1-2, t2-2] +
                    rand(dist_error)
    end
  end

  return @views mat[(prerun+1):end, (prerun+1):end]

end

# Method to fill matrix for SAR(2,2) with additive outliers
function fill_mat_dgp_sop!(
  dgp::SAR22, dist_error::ContinuousUnivariateDistribution, dist_ao::UnivariateDistribution,
  mat, mat_ao::Matrix{Float64}, mat_ma::Matrix{Float64}
)

  # Extract parameters
  prerun = dgp.prerun
  α₁ = dgp.dgp_params[1]
  α₂ = dgp.dgp_params[2]
  α₃ = dgp.dgp_params[3]
  α₄ = dgp.dgp_params[4]
  α₅ = dgp.dgp_params[5]
  α₆ = dgp.dgp_params[6]
  α₇ = dgp.dgp_params[7]
  α₈ = dgp.dgp_params[8]

  # Fill
  for t2 in 3:size(mat, 2)
    for t1 in 3:size(mat, 1)
      mat[t1, t2] = α₁ * mat[t1-1, t2] +
                    α₂ * mat[t1, t2-1] +
                    α₃ * mat[t1-1, t2-1] +
                    α₄ * mat[t1-2, t2] +
                    α₅ * mat[t1, t2-2] +
                    α₆ * mat[t1-2, t2-1] +
                    α₇ * mat[t1-1, t2-2] +
                    α₈ * mat[t1-2, t2-2] +
                    rand(dist_error)
    end
  end

  # Add AOs
  rand!(dist_ao, mat_ao)
  mat .= mat .+ mat_ao

  return @views mat[(prerun+1):end, (prerun+1):end]

end

# Method to fill matrix for SINAR(1,1) without additive outliers
function fill_mat_dgp_sop!(
  dgp::SINAR11, dist_error::DiscreteUnivariateDistribution, dist_ao::Nothing, mat,
  mat_ao::Matrix{Float64}, mat_ma::Matrix{Float64}
)

  # Extract parameters
  prerun = dgp.prerun
  α₁ = dgp.dgp_params[1]
  α₂ = dgp.dgp_params[2]
  α₃ = dgp.dgp_params[3]

  # Fill
  for t2 in 2:size(mat, 2)
    for t1 in 2:size(mat, 1)
      mat[t1, t2] = rand(Binomial(round(mat[t1-1, t2]), α₁)) +
                    rand(Binomial(round(mat[t1, t2-1]), α₂)) +
                    rand(Binomial(round(mat[t1-1, t2-1]), α₃)) +
                    rand(dist_error)
    end
  end

  return @views mat[(prerun+1):end, (prerun+1):end]

end

# Method to fill matrix for SINAR(1,1) with additive outliers
function fill_mat_dgp_sop!(
  dgp::SINAR11, dist_error::DiscreteUnivariateDistribution, dist_ao::DiscreteUnivariateDistribution,
  mat, mat_ao::Matrix{Float64}, mat_ma::Matrix{Float64}
)

  # Extract parameters
  prerun = dgp.prerun
  α₁ = dgp.dgp_params[1]
  α₂ = dgp.dgp_params[2]
  α₃ = dgp.dgp_params[3]

  # Fill
  for t2 in 2:size(mat, 2)
    for t1 in 2:size(mat, 1)
      mat[t1, t2] = rand(Binomial(round(mat[t1-1, t2]), α₁)) +
                    rand(Binomial(round(mat[t1, t2-1]), α₂)) +
                    rand(Binomial(round(mat[t1-1, t2-1]), α₃)) +
                    rand(dist_error)
    end
  end

  # Add AOs 
  rand!(dist_ao, mat_ao)
  mat .= mat .+ mat_ao

  return @views mat[(prerun+1):end, (prerun+1):end]

end

# Method to fill matrix for SQMA(1, 1) without additive outliers
function fill_mat_dgp_sop!(
  dgp::SQMA11, dist_error::ContinuousUnivariateDistribution, dist_ao::Nothing,
  mat, mat_ao::Matrix{Float64}, mat_ma::Matrix{Float64}
)

  # Extract parameters  
  prerun = dgp.prerun
  β₁ = dgp.dgp_params[1]
  β₂ = dgp.dgp_params[2]
  β₃ = dgp.dgp_params[3]

  a = dgp.eps_params[1]
  b = dgp.eps_params[2]
  c = dgp.eps_params[3]

  # draw MA-errors
  rand!(dist_error, mat_ma)

  # Fill
  for t2 in 2:size(mat, 2)
    for t1 in 2:size(mat, 1)
      mat[t1, t2] = β₁ * mat_ma[t1-1, t2]^a +
                    β₂ * mat_ma[t1, t2-1]^b +
                    β₃ * mat_ma[t1-1, t2-1]^c +
                    mat_ma[t1, t2]
    end
  end

  return @views mat[(prerun+1):end, (prerun+1):end]

end

# Method to fill matrix for SQMA(2, 2) without additive outliers
function fill_mat_dgp_sop!(
  dgp::SQMA22, dist_error::ContinuousUnivariateDistribution, dist_ao::Nothing,
  mat, mat_ao::Matrix{Float64}, mat_ma::Matrix{Float64}
)

  # Extract parameters  
  prerun = dgp.prerun
  β₁ = dgp.dgp_params[1]
  β₂ = dgp.dgp_params[2]
  β₃ = dgp.dgp_params[3]
  β₄ = dgp.dgp_params[4]
  β₅ = dgp.dgp_params[5]
  β₆ = dgp.dgp_params[6]
  β₇ = dgp.dgp_params[7]
  β₈ = dgp.dgp_params[8]

  a = dgp.eps_params[1]
  b = dgp.eps_params[2]
  c = dgp.eps_params[3]
  d = dgp.eps_params[4]
  e = dgp.eps_params[5]
  f = dgp.eps_params[6]
  g = dgp.eps_params[7]
  h = dgp.eps_params[8]

  # draw MA-errors
  rand!(dist_error, mat_ma)

  # Fill
  for t2 in 3:size(mat, 2)
    for t1 in 3:size(mat, 1)
      mat[t1, t2] = β₁ * mat_ma[t1-1, t2]^a +
                    β₂ * mat_ma[t1, t2-1]^b +
                    β₃ * mat_ma[t1-1, t2-1]^c +
                    β₄ * mat_ma[t1-2, t2]^d +
                    β₅ * mat_ma[t1, t2-2]^e +
                    β₆ * mat_ma[t1-2, t2-1]^f +
                    β₇ * mat_ma[t1-1, t2-2]^g +
                    β₈ * mat_ma[t1-2, t2-2]^h +
                    mat_ma[t1, t2]
    end
  end

  return @views mat[(prerun+1):end, (prerun+1):end]

end

# Method to fill matrix for SQINMA(1, 1) without additive outliers
function fill_mat_dgp_sop!(
  dgp::SQINMA11, dist_error::DiscreteUnivariateDistribution, dist_ao::Nothing,
  mat, mat_ao::Matrix{Float64}, mat_ma::Matrix{Float64}
)

  # Extract parameters
  prerun = dgp.prerun
  β₁ = dgp.dgp_params[1]
  β₂ = dgp.dgp_params[2]
  β₃ = dgp.dgp_params[3]

  a = dgp.eps_params[1]
  b = dgp.eps_params[2]
  c = dgp.eps_params[3]

  # draw MA-errors
  rand!(dist_error, mat_ma)

  # Fill
  for t2 in 2:size(mat, 2)
    for t1 in 2:size(mat, 1)
      mat[t1, t2] = rand(Binomial(mat_ma[t1-1, t2]^a, β₁)) +
                    rand(Binomial(mat_ma[t1, t2-1]^b, β₂)) +
                    rand(Binomial(mat_ma[t1-1, t2-1]^c, β₃)) +
                    mat_ma[t1, t2]
    end
  end

  return @views mat[(prerun+1):end, (prerun+1):end]


end

# Method to fill matrix for BSQMA(1, 1) without additive outliers
function fill_mat_dgp_sop!(
  dgp::BSQMA11, dist_error::ContinuousUnivariateDistribution, dist_ao::Nothing,
  mat, mat_ao::Matrix{Float64}, mat_ma::Matrix{Float64}
)

  # extract parameters
  prerun = dgp.prerun
  b1 = dgp.dgp_params[1]
  b2 = dgp.dgp_params[2]
  b3 = dgp.dgp_params[3]
  b4 = dgp.dgp_params[4]

  a = dgp.eps_params[1]
  b = dgp.eps_params[2]
  c = dgp.eps_params[3]
  d = dgp.eps_params[4]

  # draw ma-errors
  rand!(dist_error, mat_ma)

  # fill mat
  for t2 in 2:size(mat, 2)
    for t1 in 2:size(mat, 1)
      mat[t1, t2] = b1 * mat_ma[t1-1, t2-1]^a +
                    b2 * mat_ma[t1+1, t2-1]^b +
                    b3 * mat_ma[t1+1, t2+1]^c +
                    b4 * mat_ma[t1-1, t2+1]^d +
                    mat_ma[t1, t2]
    end
  end

  return @views mat[(prerun+1):end, (prerun+1):end]


end

