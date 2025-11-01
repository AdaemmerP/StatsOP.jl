function stat_acf(data, lam, cl, p_reps, acf_dgp, acf_dgp_dist, acf_version)

  # Pre-allocate 
  rls = Vector{Int64}(undef, length(p_reps))
  x_vec = Vector{Float64}(undef, 2)

  for r in 1:length(p_reps)

    # initialize values
    # Convert all values to ensure type stability
    if acf_version == 1
      rl = 0
      c_0 = 0.0
      s_0 = var(acf_dgp_dist)
      m_0 = mean(acf_dgp_dist)
      acf_stat = 0.0
      μ0 = mean(acf_dgp_dist)
      σ0 = std(acf_dgp_dist)

    elseif acf_version == 2
      rl = 0
      c_0 = mean(acf_dgp_dist)^2
      s_0 = var(acf_dgp_dist) + mean(acf_dgp_dist)^2
      m_0 = mean(acf_dgp_dist)
      acf_stat = 0.0
      μ0 = mean(acf_dgp_dist)
      σ0 = std(acf_dgp_dist)

    elseif acf_version == 3
      rl = 0
      c_0 = 0.0
      # --- not necessary but still ensure type stability
      s_0 = 0.0
      m_0 = 0.0
      acf_stat = 0.0
      μ0 = mean(acf_dgp_dist)
      σ0 = std(acf_dgp_dist)

    end

    # initialize sequence depending on DGP
    init_dgp_op!(acf_dgp, x_vec, acf_dgp_dist, 1)

    # set ACF statistic to zero
    acf_stat = 0

    while abs(acf_stat) < cl

      # increase run length
      rl += 1

      # compute EWMA ACF
      if acf_version == 1

        # Equation (3), page 3 in the paper
        c_0 = lam * (x_vec[2] - μ0) * (x_vec[1] - μ0) + (1.0 - lam) * c_0
        s_0 = lam * (x_vec[2] - μ0)^2 + (1.0 - lam) * s_0
        acf_stat = c_0 / s_0

      elseif acf_version == 2
        # Equation (4), page 3 in the paper
        c_0 = lam * x_vec[2] * x_vec[1] + (1.0 - lam) * c_0
        s_0 = lam * x_vec[2]^2 + (1.0 - lam) * s_0
        m_0 = lam * x_vec[2] + (1.0 - lam) * m_0
        acf_stat = (c_0 - m_0^2) / (s_0 - m_0^2)

      elseif acf_version == 3
        # Equation (5), page 3 in the paper
        c_0 = lam * (x_vec[2] - μ0) * (x_vec[1] - μ0) + (1 - lam) * c_0
        acf_stat = c_0 / σ0^2

      end


      # update x_vec depending on DGP
      update_dgp_op!(acf_dgp, x_vec, acf_dgp_dist, 1)
    end

    rls[r] = rl
  end
  return rls
end
