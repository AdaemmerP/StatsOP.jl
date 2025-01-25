using OrdinalPatterns
using RCall
using CairoMakie

R"""
library(textile)
library(spc4sts)
images_ic = textile::icImgs
images_oc = textile::ocImgs

images = textile::icImgs
model = readRDS("docs/replications/model_bui.rds")

# In-control
resid_ic_mat = array(NA, c(235, 220, 94))
for (i in 1:94) {

  dat <- dataPrep(textile::icImgs[, , i], model$nb)
  r0j <- dat[, 1] - predict(model$fit, dat)
  res <- matrix(r0j, nrow(textile::icImgs[, , i]) - model$nb[1],
               ncol(textile::icImgs[, , i]) - sum(model$nb[2:3]), byrow=TRUE)
  resid_ic_mat[, , i] = res
}

# Out-of control
resid_oc_mat = array(NA, c(235, 220, 6))
for (i in 1:6) {
  
  dat <- dataPrep(textile::ocImgs[, , i], model$nb)
  r0j <- dat[, 1] - predict(model$fit, dat)
  res <- matrix(r0j, nrow(textile::ocImgs[, , i]) - model$nb[1],
                ncol(textile::ocImgs[, , i]) - sum(model$nb[2:3]), byrow=TRUE)
  resid_oc_mat[, , i] = res
}
resid_oc_mat
"""

@rget images_ic images_oc resid_ic_mat resid_oc_mat;
images_all = cat(images_ic, images_oc, dims=3)

lam = [0.1, 1]
reps = 10^6
chart_choice = 3
L0 = 20
clinit = 0.001

# -----------------------------------------------------------------------------
#     SOP chart combined plots (Shewart and EWMA) based on original images
# -----------------------------------------------------------------------------

fig = Figure()
for i in 1:2
  cl = cl_sop(
    images_ic, lam[i], L0, clinit, 1, 1, reps;
    chart_choice=chart_choice, jmin=3, jmax=7, verbose=true
  )
  p_array = compute_p_array(images_ic, 1, 1; chart_choice=chart_choice)

  p_array_mean = mean(p_array, dims=1)
  p_array_mean = permutedims(p_array_mean, (2, 1))

  stats_sop = stat_sop(
    images_oc,
    lam[i],
    1,
    1;
    type_freq_init=p_array_mean
  )

  cl_low = chart_stat_sop(p_array_mean, chart_choice) - cl
  cl_high = chart_stat_sop(p_array_mean, chart_choice) + cl


  # Use Makie to plots stats_bp. Draw in red the critical limits  
  ax = Axis(fig[1, i], width=150, height=150)
  lines!(ax, stats_sop, color=:blue)
  scatter!(ax, stats_sop, color=:blue)
  hlines!(ax, cl_low, color=:red)
  hlines!(ax, cl_high, color=:red)
end

resize_to_layout!(fig)
fig

# -----------------------------------------------------------------------------
#     SOP chart combined plots (Shewart and EWMA) based on residuals
# -----------------------------------------------------------------------------
fig = Figure()
for i in 1:2
  cl = cl_sop(
    resid_ic_mat, lam[i], L0, clinit, 1, 1, reps;
    chart_choice=chart_choice, jmin=3, jmax=7, verbose=true
  )
  p_array = compute_p_array(resid_ic_mat, 1, 1; chart_choice=chart_choice)

  p_array_mean = mean(p_array, dims=1)
  p_array_mean = permutedims(p_array_mean, (2, 1))

  stats_sop = stat_sop(
    resid_oc_mat,
    lam[i],
    1,
    1;
    type_freq_init=p_array_mean
  )

  cl_low = chart_stat_sop(p_array_mean, chart_choice) - cl
  cl_high = chart_stat_sop(p_array_mean, chart_choice) + cl


  # Use Makie to plots stats_bp. Draw in red the critical limits
  ax = Axis(fig[1, i], width=150, height=150)
  lines!(ax, stats_sop, color=:blue)
  scatter!(ax, stats_sop, color=:blue)
  hlines!(ax, cl_low, color=:red)
  hlines!(ax, cl_high, color=:red)
end

resize_to_layout!(fig)
fig


# -----------------------------------------------------------------------------
#     BP statsitics (Shewart and EWMA) based on original images
# ----------------------------------------------------------------------------- 
w = 5

fig = Figure()
for i in 1:2
  for w in 1:w
    # Compute critical limits for BP-statistic
    cl = cl_sop_bp(
      images_ic, lam[i], 20, 7.47162523909521e-7, w, reps;
      chart_choice=3, jmin=6, jmax=9, verbose=true
    )

    # Compute in-control values    
    p_array = compute_p_array_bp(images_ic, w; chart_choice=chart_choice) # Compute relative frequencies for p-vectors
    p_array_mean = mean(p_array, dims=1)
    # Make column vectors to be compatible with p_ewma_all
    p_array_mean = permutedims(p_array_mean, (2, 1, 3))

    # Compute in-control values for test statitic
    stat_ic = zeros(size(p_array_mean, 3)) # third dimension is number of d1-d2 combinations

    for i in axes(p_array_mean, 3)
      @views stat_ic[i] = chart_stat_sop(p_array_mean[:, :, i], chart_choice)
    end

    stats_bp = stat_sop_bp(
      images_oc,
      lam[i],
      w,
      chart_choice=3,
      add_noise=false,
      stat_ic=stat_ic,
      type_freq_init=p_array_mean
    )

    # Use Makie to plots stats_bp. Draw in red the critical limits
    ax = Axis(fig[i, w], width=150, height=150)
    lines!(ax, stats_bp, color=:blue)
    scatter!(ax, stats_bp, color=:blue)
    hlines!(ax, [cl], color=:red)

  end
end
resize_to_layout!(fig)
fig

# -----------------------------------------------------------------------------
#           BP statsitics (Shewart and EWMA) based on residuals
# ----------------------------------------------------------------------------- 
w = 5

fig = Figure()
for i in 1:2
  for w in 1:w
    # Compute critical limits for BP-statistic
    cl = cl_sop_bp(
      resid_ic_mat, lam[i], 20, 7.47162523909521e-7, w, reps;
      chart_choice=3, jmin=6, jmax=9, verbose=true
    )

    # Compute in-control values    
    p_array = compute_p_array_bp(resid_ic_mat, w; chart_choice=chart_choice) # Compute relative frequencies for p-vectors
    p_array_mean = mean(p_array, dims=1)
    # Make column vectors to be compatible with p_ewma_all
    p_array_mean = permutedims(p_array_mean, (2, 1, 3))

    # Compute in-control values for test statitic
    stat_ic = zeros(size(p_array_mean, 3)) # third dimension is number of d1-d2 combinations

    for i in axes(p_array_mean, 3)
      @views stat_ic[i] = chart_stat_sop(p_array_mean[:, :, i], chart_choice)
    end

    stats_bp = stat_sop_bp(
      resid_oc_mat,
      lam[i],
      w,
      chart_choice=3,
      add_noise=false,
      stat_ic=stat_ic,
      type_freq_init=p_array_mean
    )

    # Use Makie to plots stats_bp. Draw in red the critical limits
    ax = Axis(fig[i, w], width=150, height=150)
    lines!(ax, stats_bp, color=:blue)
    scatter!(ax, stats_bp, color=:blue)
    hlines!(ax, [cl], color=:red)

  end
end
resize_to_layout!(fig)
fig
