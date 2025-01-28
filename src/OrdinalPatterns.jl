module OrdinalPatterns

# Packages to use
using Random
using LinearAlgebra
using Statistics
using Combinatorics
using Distributions
using Distributed
using StaticArrays
using StatsBase
using Reexport

# Reexport
@reexport using Distributions

# ---------------------------------------------#
# SOP related functions and structs  to export #
# ---------------------------------------------#

# Helper functions
export order_vec!,
  sop_frequencies,
  sop_frequencies!

# SOP functions to export
export arl_sop_ic,
  arl_sop_oc,
  arl_sop_bootstrap,  
  arl_sop_bp_ic,
  arl_sop_bp_oc,
  arl_sop_bp_bootstrap,
  cl_sop_ic,
  cl_sop_bootstrap,
  cl_sop_bp_ic,
  cl_sop_bp_oc,
  cl_sop_bp_bootstrap,
  stat_sop,
  stat_sop_bp,
  chart_stat_sop,
  compute_lookup_array_sop,
  compute_p_array,
  compute_p_array_bp,
  init_vals_sop,
  crit_val_sop

# SACF functions to export
export rl_sacf,
  arl_sacf,
  arl_sacf_bp,
  stat_sacf,
  stat_sacf_bp,
  sacf,
  crit_sacf,
  cl_sacf,
  cl_sacf_bp,
  crit_val_sacf

# Export functions to compute DGPs
export init_mat!,
  fill_mat_dgp_sop!,
  build_sar1_matrix,
  stat_sop

# SOP Types (from 'sop_dgp_structs.jl')
export ICSP,
  SAR11,
  SAR22,
  SINAR11,
  SQMA11,
  SQMA22,
  SQINMA11,
  SAR1,
  BSQMA11

# Custom distributions to export (from 'sop_distributions.jl')
export BinomialC,
  BinomialCvec,
  ZIP,
  PoiBin,
  BinNorm

# ---------------------------------------------#
# OP related functions and structs  to export  #
# ---------------------------------------------#

# ACF functions to export (from 'op_acf_functions.jl')
export rl_acf,
  arl_acf

# OP Types to export (from 'op_dgp_structs.jl')  
export IC,
  AR1,
  TEAR1

# OP functions to export
export rl_op,
  arl_op,
  stat_op,
  cl_op,
  count_uv_op,
  count_mv_op,
  dependence_op,
  changepoint_op

# ---------------------------------------------#
#  OP related functions and structs to include #  
# ---------------------------------------------#
# OP files
include("op/op_dgp_structs.jl")
include("op/op_arl_functions.jl")
include("op/op_dependence.jl")
include("op/op_help_functions.jl")
include("op/op_stat_functions.jl")
include("op/op_test_functions.jl")
include("op/op_dgp_functions.jl")

# ACF files 
include("acf/op_acf_functions.jl")

# ---------------------------------------------#
# SOP related functions and structs to include #  
# ---------------------------------------------#
# ---
include("sop/sop_dgp_structs.jl")
# ---
include("sop/sop_arl_ic_functions.jl")
include("sop/sop_arl_oc_functions.jl")
include("sop/sop_arl_bootstrap_functions.jl")
# ---
include("sop/sop_bp_arl_ic_functions.jl")
include("sop/sop_bp_arl_oc_functions.jl")
include("sop/sop_bp_arl_bootstrap_functions.jl")
# ---
include("sop/sop_cl_ic_functions.jl")
include("sop/sop_cl_bootstrap_functions.jl")
include("sop/sop_cl_bp_ic.jl")
include("sop/sop_cl_bp_bootstrap.jl")
# ---
include("sop/sop_stat_functions.jl")
include("sop/sop_stat_bp_functions.jl")
include("sop/sop_dgp_functions.jl")
include("sop/sop_distributions.jl")
include("sop/sop_help_functions.jl")

# SACFs
include("sacf/sacf_arl_functions.jl")
include("sacf/sacf_arl_bp_functions.jl")
include("sacf/sacf_cl_functions.jl")
include("sacf/sacf_stat_functions.jl")
include("sacf/sacf_stat_bp_functions.jl")

end
