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
export rl_sop, 
       arl_sop,
       arl_sop_bp,
       cl_sop,
       cl_sop_bp,
       stat_sop,
       stat_sop_bp,
       chart_stat_sop,
       compute_lookup_array_sop,
       compute_p_array,
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

# SOP Types to export (from 'sop_dgp_structs.jl')
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
# OP related functions and structs  to export #
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

# Include files for OPs
include("op/op_dgp_structs.jl")
include("op/op_arl_functions.jl")
include("op/op_dependence.jl")
include("op/op_help_functions.jl")
include("op/op_stat_functions.jl")
include("op/op_test_functions.jl")
include("op/op_dgp_functions.jl")

# Include files for ACFs 
include("acf/op_acf_functions.jl")

# Include files for SOPs
include("sop/sop_dgp_structs.jl")
include("sop/sop_arl_functions.jl")
include("sop/sop_arl_bp_functions.jl")
include("sop/sop_dgp_functions.jl")
include("sop/sop_distributions.jl")
include("sop/sop_stat_functions.jl")
include("sop/sop_stat_bp_functions.jl")
include("sop/sop_help_functions.jl")
include("sop/sop_cl_functions.jl")

# Include files for SACFs
include("sacf/sacf_arl_functions.jl")
include("sacf/sacf_arl_bp_functions.jl")
include("sacf/sacf_cl_functions.jl")
include("sacf/sacf_stat_functions.jl")
include("sacf/sacf_stat_bp_functions.jl")

end
