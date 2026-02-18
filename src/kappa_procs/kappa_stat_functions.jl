export stat_kappa

function stat_kappa(
  data::Vector{T}, lam, dist_null, chart_choice::KappaN1
) where T<:Real

  # Pre-allocate variables
  # Compute support
  stat_all = zeros(length(data) - 1)
  sup_lb, sup_ub = get_bounds(dist_null)
  sup = collect(sup_lb:sup_ub)
  Bₜ = zeros(Int, length(sup))
  Bₜ₋₁ = similar(Bₜ)

  # Initialize at t = 0
  qₜ = pdf(dist_null, sup)
  Qₜ = sum(qₜ .^ 2)

  for r in 2:length(data)

    # Set match counts
    @. Bₜ = (sup == data[r])
    @. Bₜ₋₁ = (sup == data[r-1])
    dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

    # Compute EWMA statistic
    @. qₜ = lam * Bₜ + (1 - lam) * qₜ
    Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
    stat_all[r-1] = chart_stat_qual(qₜ, Qₜ, chart_choice)

    # reset match counts
    fill!(Bₜ, 0)
    fill!(Bₜ₋₁, 0)

  end
  return stat_all

end

# Function to compute D-chart and Persistence 
function stat_kappa(
  data::Vector{T}, lam, dist_null, chart_choice::KappaN2
) where T<:Real

  # Pre-allocate variables
  # Compute support
  stat_all = zeros(length(data) - 1)
  sup_lb, sup_ub = get_bounds(dist_null)
  sup = collect(sup_lb:sup_ub)
  Bₜ = zeros(Int, length(sup))
  Bₜ₋₁ = similar(Bₜ)

  # Initialize at t = 0
  p₀ = pdf(dist_null, sup)
  Qₜ = sum(p₀ .^ 2)

  for r in 2:length(data)

    # Set match counts
    @. Bₜ = (sup == data[r])
    @. Bₜ₋₁ = (sup == data[r-1])
    dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

    # Compute EWMA statistic
    Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
    stat_all[r-1] = chart_stat_qual(p₀, Qₜ, chart_choice)

    # reset match counts
    fill!(Bₜ, 0)
    fill!(Bₜ₋₁, 0)

  end
  return stat_all

end


function stat_kappa(
  data::Vector{T}, lam, dist_null, chart_choice::KappaO1
) where T<:Real

  # Pre-allocate variables
  # Compute support
  stat_all = zeros(length(data) - 1)
  sup_lb, sup_ub = get_bounds(dist_null)
  sup = collect(sup_lb:sup_ub)
  Bₜ = zeros(Int, length(sup))
  Bₜ₋₁ = similar(Bₜ)

  # Initialize at t = 0
  qₜ = cdf(dist_null, sup)
  Qₜ = sum(qₜ .^ 2)

  for r in 2:length(data)

    # Set match counts
    @. Bₜ = (sup == data[r])
    @. Bₜ₋₁ = (sup == data[r-1])
    dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

    # Compute EWMA statistic
    @. qₜ = lam * Bₜ + (1 - lam) * qₜ
    Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
    stat_all[r-1] = chart_stat_qual(qₜ, Qₜ, chart_choice)

    # reset match counts
    fill!(Bₜ, 0)
    fill!(Bₜ₋₁, 0)

  end
  return stat_all

end

# Function to compute D-chart and Persistence 
function stat_kappa(
  data::Vector{T}, lam, dist_null, chart_choice::KappaO2
) where T<:Real

  # Pre-allocate variables
  # Compute support
  stat_all = zeros(length(data) - 1)
  sup_lb, sup_ub = get_bounds(dist_null)
  sup = collect(sup_lb:sup_ub)
  Bₜ = zeros(Int, length(sup))
  Bₜ₋₁ = similar(Bₜ)

  # Initialize at t = 0
  f₀ = cdf(dist_null, sup)
  Qₜ = sum(f₀ .^ 2)

  for r in 2:length(data)

    # Set match counts
    @. Bₜ = (sup == data[r])
    @. Bₜ₋₁ = (sup == data[r-1])
    dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

    # Compute EWMA statistic
    Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
    stat_all[r-1] = chart_stat_qual(f₀, Qₜ, chart_choice)

    # reset match counts
    fill!(Bₜ, 0)
    fill!(Bₜ₋₁, 0)

  end
  return stat_all

end
