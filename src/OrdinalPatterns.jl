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
using ComplexityMeasures: InformationMeasure, ComplexityEstimator, Shannon, ShannonExtropy
import PrecompileTools

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
  cl_sop,
  cl_sop_bootstrap,
  cl_sop_bp,
  cl_sop_bp_bootstrap,
  stat_sop,
  stat_sop_bp,
  chart_stat_sop,
  compute_lookup_array_sop,
  compute_p_array,
  compute_p_array_bp,
  init_vals_sop,
  init_mat!,
  fill_mat_dgp_sop!,
  crit_val_sop

# SACF functions to export
export rl_sacf,
  arl_sacf_ic,
  arl_sacf_oc,
  arl_sacf_bp_ic,
  arl_sacf_bp_oc,
  stat_sacf,
  stat_sacf_bp,
  sacf,
  cl_sacf,
  cl_sacf_bp,
  test_sop,
  crit_val_sacf

# Export functions to compute DGPs
export init_mat!,
  fill_mat_dgp_sop!,
  build_sar1_matrix,
  stat_sop

# SOP Types (from 'sop_dgp_structs.jl')
export ICSTS,
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
export ICTS,
  AR1,
  TEAR1,
  MA1,
  MA2

# OP functions to export
export rl_op_ic,
  rl_op_oc,
  arl_op_ic,
  arl_op_oc,
  stat_op,
  test_op,
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
include("op/op_arl_ic_functions.jl")
include("op/op_arl_oc_functions.jl")
include("op/op_cl_functions.jl")
include("op/op_dependence.jl")
include("op/op_help_functions.jl")
include("op/op_stat_functions.jl")
include("op/op_test_functions.jl")
include("op/op_dgp_functions.jl")

# ACF files 
include("acf/acf_functions.jl")
include("acf/acf_cl_functions.jl")

# ---------------------------------------------#
# SOP related functions and structs to include #  
# ---------------------------------------------#
# ---
include("sop/information_measures.jl")
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
include("sop/sop_cl_functions.jl")
include("sop/sop_cl_bootstrap_functions.jl")
include("sop/sop_bp_cl_functions.jl")
include("sop/sop_bp_cl_bootstrap_functions.jl")
# ---
include("sop/sop_stat_functions.jl")
include("sop/sop_stat_bp_functions.jl")
include("sop/sop_dgp_functions.jl")
include("sop/sop_distributions.jl")
include("sop/sop_help_functions.jl")
include("sop/sop_test_functions.jl")

# SACFs
include("sacf/sacf_arl_ic_functions.jl")
include("sacf/sacf_arl_oc_functions.jl")
include("sacf/sacf_bp_arl_ic_functions.jl")
include("sacf/sacf_bp_arl_oc_functions.jl")
include("sacf/sacf_cl_functions.jl")
include("sacf/sacf_cl_bp_functions.jl")
include("sacf/sacf_stat_functions.jl")
include("sacf/sacf_stat_bp_functions.jl")

# Precompile
#include("other/precompile.jl")

end
