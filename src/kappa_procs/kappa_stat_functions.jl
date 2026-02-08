export stat_kappa

function stat_kappa(
  data::Vector{Real}, lam, null_dist, chart_choice::KappaN1
)

  # Pre-allocate variables
  stat_all = zeros(length(data) - 1)
  sup = support(null_dist)
  Bₜ = zeros(Int, length(sup))
  Bₜ₋₁ = similar(Bₜ)

  # Initialize at t = 0
  qₜ = pdf(null_dist)
  Qₜ = sum(qₜ .^ 2)

  for r in 2:length(data)-1

    # Set match counts
    @. Bₜ = (sup == data[r])
    @. Bₜ₋₁ = (sup == data[r-1])
    # Update
    @. qₜ = lam * Bₜ + (1 - lam) * qₜ
    dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

    # Compute EWMA statistic
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
  data::Vector{Real}, lam, null_dist, chart_choice::KappaN2
)

  # Pre-allocate variables
  stat_all = zeros(length(data) - 1)
  sup = support(null_dist)
  Bₜ = zeros(Int, length(sup))
  Bₜ₋₁ = similar(Bₜ)

  # Initialize at t = 0
  p₀ = pdf(qual_dgp_dist)
  Qₜ = sum(p₀ .^ 2)

  for r in 2:length(data)-1

    # Set match counts
    @. Bₜ = (sup == data[r])
    @. Bₜ₋₁ = (sup == data[r-1])
    # Update
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
  data::Vector{Real}, lam, null_dist, chart_choice::KappaO1
)

  # Pre-allocate variables
  stat_all = zeros(length(data) - 1)
  sup = support(null_dist)
  Bₜ = zeros(Int, length(sup))
  Bₜ₋₁ = similar(Bₜ)

  # Initialize at t = 0
  qₜ = cdf(null_dist, sup)
  Qₜ = sum(qₜ .^ 2)

  for r in 2:length(data)-1

    # Set match counts
    @. Bₜ = (sup == data[r])
    @. Bₜ₋₁ = (sup == data[r-1])
    # Update
    @. qₜ = lam * Bₜ + (1 - lam) * qₜ
    dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

    # Compute EWMA statistic
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
  data::Vector{Real}, lam, null_dist, chart_choice::KappaO2
)

  # Pre-allocate variables
  stat_all = zeros(length(data) - 1)
  sup = support(null_dist)
  Bₜ = zeros(Int, length(sup))
  Bₜ₋₁ = similar(Bₜ)

  # Initialize at t = 0
  f₀ = cdf(null_dist, sup)
  Qₜ = sum(f₀ .^ 2)

  for r in 2:length(data)-1

    # Set match counts
    @. Bₜ = (sup == data[r])
    @. Bₜ₋₁ = (sup == data[r-1])
    # Update
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
