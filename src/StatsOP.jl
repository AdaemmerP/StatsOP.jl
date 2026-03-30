module StatsOP

# Packages to use
using Combinatorics
using ComplexityMeasures: InformationMeasure, ComplexityEstimator, Entropy, Shannon, ShannonExtropy
using Distributions
using GeneralizedChisqDistribution
using LinearAlgebra
using Random
using Reexport
using Roots
using StaticArrays
using Statistics
using StatsBase
import PrecompileTools

# Reexport
@reexport using Distributions
@reexport using ComplexityMeasures: Shannon, ShannonExtropy

# -----------------------------------------------#
#  General helper functions and structs to export #
# -----------------------------------------------#

# algorithms_and_types/lehmer_function.jl
export perm_to_lehm_idx,
  perm_to_lehm_idx!

# ---------------------------------------------#
# OP related functions and structs  to export  #
# ---------------------------------------------#

# op/op_dgp_structs.jl
export AR1,
  BAR1,
  ContinuousDGPIC,
  DAR1,
  DiscreteDGPIC,
  INAR1,
  MA1,
  MA2,
  QAR1,
  SINAR1,
  TEAR1,
  TINAR1,
  WDAR1

# op/op_information_measures.jl
export chart_stat_op,
  DistanceToWhiteNoise,
  Persistence,
  RotationalAsymmetry,
  UpDownBalance,
  UpDownScaling

# op/op_stat_bp_functions.jl
export crit_val_op_bp,
  stat_op_bp

# op/op_help_functions.jl
export add_noise!

# op/op_arl_ic_functions.jl, op_arl_oc_functions.jl, op_cl_functions.jl,
# op_dependence.jl, op_stat_functions.jl, op_test_functions.jl
export arl_op_ic,
  arl_op_oc,
  changepoint_op,
  cl_op,
  count_mv_op,
  count_uv_op,
  dependence_op,
  rl_op_ic,
  rl_op_oc,
  stat_op,
  test_op

# ---------------------------------------------#
# ACF related functions and structs to export  #
# ---------------------------------------------#

# acf/acf_arl_ic_functions.jl
export arl_acf,
  rl_acf

# acf/acf_arl_oc_functions.jl
export arl_acf_oc

# acf/acf_cl_functions.jl
export cl_acf

# ----------------------------------------------#
#  GOP related functions and structs to export  #
# ----------------------------------------------#

# gop/gop_information_measures.jl
export chart_stat_gop,
  D_Chart

# gop/gop_arl_ic_functions.jl
export arl_gop_ic,
  rl_gop_ic

# gop/gop_arl_oc_functions.jl
export arl_gop_oc,
  rl_gop_oc

# gop/gop_cl_functions.jl
export cl_gop

# gop/gop_help_functions.jl
export abort_criterium_gop,
  competerank!,
  compute_lookup_array_gop,
  fill_p0!

# gop/gop_stat_functions.jl
export stat_gop

# ---------------------------------------------#
# SOP related functions and structs to export  #
# ---------------------------------------------#

# sop/sop_information_measures.jl
export chart_stat_sop,
  DiagonalType,
  DirectionType,
  KappaHat,
  KappaTilde,
  RefinedType,
  RotationType,
  TauHat,
  TauTilde

# sop/sop_dgp_structs.jl
export BSQMA11,
  ICSTS,
  SAR1,
  SAR11,
  SAR22,
  SINAR11,
  SQINMA11,
  SQMA11,
  SQMA22

# sop/sop_distributions.jl
export BinNorm,
  BinomialC,
  BinomialCvec,
  PoiBin,
  ZIP

# sop/sop_dgp_functions.jl
export build_sar1_matrix,
  fill_mat_dgp_sop!,
  init_mat!

# sop/sop_help_functions.jl
export compute_lookup_array_sop,
  compute_p_array,
  compute_p_array_bp,
  sop_frequencies!

# sop/sop_arl_ic_functions.jl
export arl_sop_ic

# sop/sop_arl_oc_functions.jl
export arl_sop_oc

# sop/sop_arl_bootstrap_functions.jl
export arl_sop_bootstrap

# sop/sop_bp_arl_ic_functions.jl
export arl_sop_bp_ic

# sop/sop_bp_arl_oc_functions.jl
export arl_sop_bp_oc

# sop/sop_bp_arl_bootstrap_functions.jl
export arl_sop_bp_bootstrap

# sop/sop_cl_functions.jl
export cl_sop

# sop/sop_cl_bootstrap_functions.jl
export cl_sop_bootstrap

# sop/sop_bp_cl_functions.jl
export cl_sop_bp

# sop/sop_bp_cl_bootstrap_functions.jl
export cl_sop_bp_bootstrap

# sop/sop_stat_functions.jl
export stat_sop

# sop/sop_stat_bp_functions.jl
export stat_sop_bp

# sop/sop_test_functions.jl
export crit_val_sop,
  test_sop

# ---------------------------------------------#
# SACF related functions and structs to export #
# ---------------------------------------------#

# sacf/sacf_arl_ic_functions.jl
export arl_sacf_ic

# sacf/sacf_arl_oc_functions.jl
export arl_sacf_oc

# sacf/sacf_bp_arl_ic_functions.jl
export arl_sacf_bp_ic

# sacf/sacf_bp_arl_oc_functions.jl
export arl_sacf_bp_oc

# sacf/sacf_cl_functions.jl
export cl_sacf

# sacf/sacf_cl_bp_functions.jl
export cl_sacf_bp

# sacf/sacf_stat_functions.jl
export crit_val_sacf,
  sacf,
  stat_sacf

# sacf/sacf_stat_bp_functions.jl
export stat_sacf_bp

# ---------------------------------------------#
# Kappa related functions and structs to export#
# ---------------------------------------------#

# kappa_procs/kappa_information_measures.jl
export chart_stat_qual,
  KappaN,
  KappaN1,
  KappaN2,
  KappaO,
  KappaO1,
  KappaO2

# kappa_procs/kappa_arl_ic_functions.jl
export arl_kappa_ic,
  rl_kappa_ic

# kappa_procs/kappa_arl_oc_functions.jl
export arl_kappa_oc,
  rl_kappa_oc

# kappa_procs/kappa_cl_functions.jl
export cl_kappa

# kappa_procs/kappa_stat_functions.jl
export stat_kappa


# -----------------------------------------------#
#  General helper functions and structs to include#
# -----------------------------------------------#
include("algorithms_and_types/lehmer_function.jl")

# ---------------------------------------------#
#  OP related functions and structs to include #
# ---------------------------------------------#
include("op/op_dgp_structs.jl")
include("op/op_dgp_functions.jl")
include("op/op_information_measures.jl")
include("op/op_arl_ic_functions.jl")
include("op/op_arl_oc_functions.jl")
include("op/op_cl_functions.jl")
include("op/op_dependence.jl")
include("op/op_stat_functions.jl")
include("op/op_stat_bp_functions.jl")
include("op/op_test_functions.jl")
include("op/op_help_functions.jl")

# ACF files
include("acf/acf_arl_ic_functions.jl")
include("acf/acf_arl_oc_functions.jl")
include("acf/acf_cl_functions.jl")

# ----------------------------------------------#
#  GOP related functions and structs to include #
# ----------------------------------------------#
include("gop/gop_information_measures.jl")
include("gop/gop_arl_ic_functions.jl")
include("gop/gop_arl_oc_functions.jl")
include("gop/gop_cl_functions.jl")
include("gop/gop_help_functions.jl")
include("gop/gop_stat_functions.jl")

# ---------------------------------------------#
# SOP related functions and structs to include #
# ---------------------------------------------#
include("sop/sop_information_measures.jl")
include("sop/sop_dgp_structs.jl")
include("sop/sop_dgp_functions.jl")
include("sop/sop_distributions.jl")
include("sop/sop_arl_ic_functions.jl")
include("sop/sop_arl_oc_functions.jl")
include("sop/sop_arl_bootstrap_functions.jl")
include("sop/sop_help_functions.jl")
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

# Kappa scripts
include("kappa_procs/kappa_information_measures.jl")
include("kappa_procs/kappa_arl_ic_functions.jl")
include("kappa_procs/kappa_arl_oc_functions.jl")
include("kappa_procs/kappa_cl_functions.jl")
include("kappa_procs/kappa_stat_functions.jl")


# Precompile
#include("other/precompile.jl")

end
