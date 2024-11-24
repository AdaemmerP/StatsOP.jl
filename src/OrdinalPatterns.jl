module OrdinalPatterns

# Packages to use
using Random
using LinearAlgebra
using Statistics
using Combinatorics
using Distributions
using Distributed

# Helper functions
export order_vec!,
       sop_frequencies,
       sop_frequencies!

# SOP functions to export
export rl_sop, 
       arl_sop,
       cl_sop,
       stat_sop,
       chart_stat_sop,
       compute_lookup_array,
       compute_p_mat,
       init_vals_sop,
       crit_val_sop

# SACF functions to export
export rl_sacf,
       arl_sacf,
       sacf_11,
       crit_sacf,
       cl_sacf,
       crit_val_sacf    

# Export functions to compute DGPs
export init_mat!,
       fill_mat_dgp_sop!,
       build_sar1_matrix,
       stat_sop

# SOP DGP-structs to export (from 'sop_dgp_structs.jl')
export SAR11,
       SINAR11,
       SQMA11,
       SQINMA11,
       SAR1,
       BSQMA11 

# Custom distributions to export (from 'sop_distributions.jl')
export BinomialC,
       BinomialCvec,
       ZIP,
       PoiBin,
       BinNorm

# Include files
include("sop_arl_functions.jl")
include("sop_dgp_structs.jl")
include("sop_dgp_functions.jl")
include("sop_distributions.jl")
include("sop_test_functions.jl")
include("sacf_functions.jl")


end
