
cd(@__DIR__)
using JLD2
using LatexPrint
using RCall

# ----------------------------------------------------------------------#
# ---------------------      In_Control            ---------------------# 
# ----------------------------------------------------------------------#

# SACF in-control ARLs
arl_ic_sacf_delays = load_object("ic/arl_ic_sacf_delays.jld2")
@rput arl_ic_sacf_delays

# Merge
arl_ic_sacf_delays = reshape(arl_ic_sacf_delays, 4, 9)
@rput arl_ic_sacf_delays
R"""
library(xtable)
arl_ic_sacf_delays_tmp <- round(arl_ic_sacf_delays, 3)
colnames(arl_ic_sacf_delays_tmp) <- rep(c("$d_1, d_2=1$", "$d_1, d_2=2$", "$d_1, d_2=3$"), 3)
rownames(arl_ic_sacf_delays_tmp) <- c("$10, 10$", "$15,15$", "$25,25$", "$40,25$")
print(xtable(arl_ic_sacf_delays_tmp, digits=2, caption = "SACF-IC-Results for Normal (0, 1), T(2), and Exponential(1)."),
      type = "latex",
      add.rownames = TRUE,
      include.colnames = TRUE,
      align = "l|lllllllll",
      caption.placement = "top",
      sanitize.rownames.function = function(x) {x},
      sanitize.colnames.function = function(x) {x}   
      )
"""

# SACF in-control ARLs
arl_ic_sacf_bp = load_object("ic/arl_ic_sacf_bp.jld2")
arl_ic_sacf_bp = reshape(arl_ic_sacf_bp, 4, 9)
@rput arl_ic_sacf_bp

R"""
library(xtable)
arl_ic_sacf_bp_tmp <- round(arl_ic_sacf_bp, 3)
colnames(arl_ic_sacf_bp_tmp) <- rep(c("$w=1$", "$w=2$", "$w=3$"), 3)
rownames(arl_ic_sacf_bp_tmp) <- c("$10, 10$", "$15,15$", "$25,25$", "$40,25$")
print(xtable(arl_ic_sacf_bp_tmp, digits=2, caption = "SACF-BP-Results for Normal (0, 1), T(2), and Exponential(1)."),
      type = "latex",
      add.rownames = TRUE,
      include.colnames = TRUE,
      align = "l|lllllllll",
      caption.placement = "top",
      sanitize.rownames.function = function(x) {x},
      sanitize.colnames.function = function(x) {x}   
      )
"""


# ----------------------------------------------------------------------#
# ---------------------      SAR(1, 1)             ---------------------# 
# ----------------------------------------------------------------------#

# Delays without outliers
arl_sop_sar11 = load_object("sar11/arl_sop_sar11.jld2")
arl_sacf_sar11 = load_object("sar11/arl_sacf_sar11.jld2")
arl_sar11_sacf_sop = hcat(arl_sop_sar11, arl_sacf_sar11)
@rput arl_sar11_sacf_sop

R"""
library(xtable)
arl_sar11_sacf_sop_tmp <- round(arl_sar11_sacf_sop, 3)
colnames(arl_sar11_sacf_sop_tmp) <- rep(c("$d_1, d_2=1$", "$d_1, d_2=2$", "$d_1, d_2=3$"), 2)
rownames(arl_sar11_sacf_sop_tmp) <- c("$10, 10$", "$15,15$", "$25,25$", "$40,25$")  
print(xtable(arl_sar11_sacf_sop_tmp, digits=2, caption = "SOP-Results and SACF-Results for SAR(1, 1)."),
      type = "latex",
      add.rownames = TRUE,
      include.colnames = TRUE,
      align = "l|lllllllll",
      caption.placement = "top",
      sanitize.rownames.function = function(x) {x},
      sanitize.colnames.function = function(x) {x}   
      )
"""


# Delays with outliers
arl_sop_sar11_outl = load_object("sar11/arl_sop_sar11_outl.jld2")
arl_sacf_sar11_outl = load_object("sar11/arl_sacf_sar11_outl.jld2")
arl_sar11_sacf_sop_outl = hcat(arl_sop_sar11_outl, arl_sacf_sar11_outl)
@rput arl_sar11_sacf_sop_outl

R"""
library(xtable)
arl_sar11_sacf_sop_outl_tmp <- round(arl_sar11_sacf_sop_outl, 3)
colnames(arl_sar11_sacf_sop_outl_tmp) <- rep(c("$d_1, d_2=1$", "$d_1, d_2=2$", "$d_1, d_2=3$"), 2)
rownames(arl_sar11_sacf_sop_outl_tmp) <- c("$10, 10$", "$15,15$", "$25,25$", "$40,25$")
print(xtable(arl_sar11_sacf_sop_outl_tmp, digits=2, caption = "SOP-Results and SACF-Results for SAR(1, 1) with outliers."),
      type = "latex",
      add.rownames = TRUE,
      include.colnames = TRUE,
      align = "l|lllllllll",
      caption.placement = "top",
      sanitize.rownames.function = function(x) {x},
      sanitize.colnames.function = function(x) {x}   
      )
"""


# BP without outliers
arl_sop_sar11_bp = load_object("sar11/arl_sop_bp_sar11.jld2")
arl_sacf_sar11_bp = load_object("sar11/arl_sacf_bp_sar11.jld2")
arl_sar11_sacf_sop_bp = hcat(arl_sop_sar11_bp, arl_sacf_sar11_bp)
@rput arl_sar11_sacf_sop_bp

R"""
library(xtable)
arl_sar11_sacf_sop_bp_tmp <- round(arl_sar11_sacf_sop_bp, 3)
colnames(arl_sar11_sacf_sop_bp_tmp) <- rep(c("$w=1$", "$w=2$", "$w=3$"), 2)
rownames(arl_sar11_sacf_sop_bp_tmp) <- c("$10, 10$", "$15,15$", "$25,25$", "$40,25$")
print(xtable(arl_sar11_sacf_sop_bp_tmp, digits=2, caption = "SOP-BP-Results and SACF-BP-Results for SAR(1, 1)."),
      type = "latex",
      add.rownames = TRUE,
      include.colnames = TRUE,
      align = "l|lllllllll",
      caption.placement = "top",
      sanitize.rownames.function = function(x) {x},
      sanitize.colnames.function = function(x) {x}   
      )
"""

# BP with outliers
arl_sop_sar11_bp_outl = load_object("sar11/arl_sop_bp_sar11_outl.jld2")
arl_sacf_sar11_bp_outl = load_object("sar11/arl_sacf_bp_sar11_outl.jld2")
arl_sar11_sacf_sop_bp_outl = hcat(arl_sop_sar11_bp_outl, arl_sacf_sar11_bp_outl)
@rput arl_sar11_sacf_sop_bp_outl

R"""
library(xtable)
arl_sar11_sacf_sop_bp_outl_tmp <- round(arl_sar11_sacf_sop_bp_outl, 3)
colnames(arl_sar11_sacf_sop_bp_outl_tmp) <- rep(c("$w=1$", "$w=2$", "$w=3$"), 2)
rownames(arl_sar11_sacf_sop_bp_outl_tmp) <- c("$10, 10$", "$15,15$", "$25,25$", "$40,25$")
print(xtable(arl_sar11_sacf_sop_bp_outl_tmp, digits=2, caption = "SOP-BP-Results and SACF-BP-Results for SAR(1, 1) with outliers."),
      type = "latex",
      add.rownames = TRUE,
      include.colnames = TRUE,
      align = "l|lllllllll",
      caption.placement = "top",
      sanitize.rownames.function = function(x) {x},
      sanitize.colnames.function = function(x) {x}   
      )
"""


# ----------------------------------------------------------------------#
# ---------------------      SAR(2, 2)             ---------------------# 
# ----------------------------------------------------------------------#

# Delays without outliers
arl_sop_sar22 = load_object("sar22/arl_sop_sar22.jld2")
arl_sacf_sar22 = load_object("sar22/arl_sacf_sar22.jld2")
arl_sar22_sacf_sop = hcat(arl_sop_sar22, arl_sacf_sar22)
@rput arl_sar22_sacf_sop

R"""
library(xtable)
arl_sar22_sacf_sop_tmp <- round(arl_sar22_sacf_sop, 3)
colnames(arl_sar22_sacf_sop_tmp) <- rep(c("$d_1, d_2=1$", "$d_1, d_2=2$", "$d_1, d_2=3$"), 2)
rownames(arl_sar22_sacf_sop_tmp) <- c("$10, 10$", "$15,15$", "$25,25$", "$40,25$")
print(xtable(arl_sar22_sacf_sop_tmp, digits=2, caption = "SOP-Results and SACF-Results for SAR(2, 2)."),
      type = "latex",
      add.rownames = TRUE,
      include.colnames = TRUE,
      align = "l|lllllllll",
      caption.placement = "top",
      sanitize.rownames.function = function(x) {x},
      sanitize.colnames.function = function(x) {x}   
      )
"""


# Delays with outliers
arl_sop_sar22_outl = load_object("sar22/arl_sop_sar22_outl.jld2")
arl_sacf_sar22_outl = load_object("sar22/arl_sacf_sar22_outl.jld2")
arl_sar22_sacf_sop_outl = hcat(arl_sop_sar22_outl, arl_sacf_sar22_outl)
@rput arl_sar22_sacf_sop_outl

R"""
library(xtable)
arl_sar22_sacf_sop_outl_tmp <- round(arl_sar22_sacf_sop_outl, 3)
colnames(arl_sar22_sacf_sop_outl_tmp) <- rep(c("$d_1, d_2=1$", "$d_1, d_2=2$", "$d_1, d_2=3$"), 2)
rownames(arl_sar22_sacf_sop_outl_tmp) <- c("$10, 10$", "$15,15$", "$25,25$", "$40,25$")
print(xtable(arl_sar22_sacf_sop_outl_tmp, digits=2, caption = "SOP-Results and SACF-Results for SAR(2, 2) with outliers."),
      type = "latex",
      add.rownames = TRUE,
      include.colnames = TRUE,
      align = "l|lllllllll",
      caption.placement = "top",
      sanitize.rownames.function = function(x) {x},
      sanitize.colnames.function = function(x) {x}   
      )
"""


# BP without outliers
arl_sop_sar22_bp = load_object("sar22/arl_sop_sar22_bp.jld2")
arl_sacf_sar22_bp = load_object("sar22/arl_sacf_sar22_bp.jld2")
arl_sar22_sacf_sop_bp = hcat(arl_sop_sar22_bp, arl_sacf_sar22_bp)
@rput arl_sar22_sacf_sop_bp

R"""
library(xtable)
arl_sar22_sacf_sop_bp_tmp <- round(arl_sar22_sacf_sop_bp, 3)
colnames(arl_sar22_sacf_sop_bp_tmp) <- rep(c("$w=1$", "$w=2$", "$w=3$"), 2)
rownames(arl_sar22_sacf_sop_bp_tmp) <- c("$10, 10$", "$15,15$", "$25,25$", "$40,25$")
print(xtable(arl_sar22_sacf_sop_bp_tmp, digits=2, caption = "SOP-BP-Results and SACF-BP-Results for SAR(2, 2)."),
      type = "latex",
      add.rownames = TRUE,
      include.colnames = TRUE,
      align = "l|lllllllll",
      caption.placement = "top",
      sanitize.rownames.function = function(x) {x},
      sanitize.colnames.function = function(x) {x}   
      )
"""


# BP with outliers
arl_sop_sar22_bp_outl = load_object("sar22/arl_sop_sar22_outl_bp.jld2")
arl_sacf_sar22_bp_outl = load_object("sar22/arl_sacf_sar22_outl_bp.jld2")
arl_sar22_sacf_sop_bp_outl = hcat(arl_sop_sar22_bp_outl, arl_sacf_sar22_bp_outl)
@rput arl_sar22_sacf_sop_bp_outl

R"""
library(xtable)
arl_sar22_sacf_sop_bp_outl_tmp <- round(arl_sar22_sacf_sop_bp_outl, 3)
colnames(arl_sar22_sacf_sop_bp_outl_tmp) <- rep(c("$w=1$", "$w=2$", "$w=3$"), 2)
rownames(arl_sar22_sacf_sop_bp_outl_tmp) <- c("$10, 10$", "$15,15$", "$25,25$", "$40,25$")
print(xtable(arl_sar22_sacf_sop_bp_outl_tmp, digits=2, caption = "SOP-BP-Results and SACF-BP-Results for SAR(2, 2) with outliers."),
      type = "latex",
      add.rownames = TRUE,
      include.colnames = TRUE,
      align = "l|lllllllll",
      caption.placement = "top",
      sanitize.rownames.function = function(x) {x},
      sanitize.colnames.function = function(x) {x}   
      )
"""

# ----------------------------------------------------------------------#
# ---------------------      SQMA(1, 1)            ---------------------# 
# ----------------------------------------------------------------------#

# Delays without outliers
arl_sop_sqma11 = load_object("sqma11/arl_sop_sqma11.jld2")
arl_sacf_sqma11 = load_object("sqma11/arl_sacf_sqma11.jld2")
arl_sqma11_sacf_sop = hcat(arl_sop_sqma11, arl_sacf_sqma11)
@rput arl_sqma11_sacf_sop

R"""
library(xtable)
arl_sqma11_sacf_sop_tmp <- round(arl_sqma11_sacf_sop, 3)
colnames(arl_sqma11_sacf_sop_tmp) <- rep(c("$d_1, d_2=1$", "$d_1, d_2=2$", "$d_1, d_2=3$"), 2)
rownames(arl_sqma11_sacf_sop_tmp) <- c("$10, 10$", "$15,15$", "$25,25$", "$40,25$")
print(xtable(arl_sqma11_sacf_sop_tmp, digits=2, caption = "SOP-Results and SACF-Results for SQMA(1, 1)."),
      type = "latex",
      add.rownames = TRUE,
      include.colnames = TRUE,
      align = "l|lllllllll",
      caption.placement = "top",
      sanitize.rownames.function = function(x) {x},
      sanitize.colnames.function = function(x) {x}   
      )
"""


# BP without outliers
arl_sop_sqma11_bp = load_object("sqma11/arl_sop_bp_sqma11.jld2")
arl_sacf_sqma11_bp = load_object("sqma11/arl_sacf_bp_sqma11.jld2")
arl_sqma11_sacf_sop_bp = hcat(arl_sop_sqma11_bp, arl_sacf_sqma11_bp)
@rput arl_sqma11_sacf_sop_bp

R"""
library(xtable)
arl_sqma11_sacf_sop_bp_tmp <- round(arl_sqma11_sacf_sop_bp, 3)
colnames(arl_sqma11_sacf_sop_bp_tmp) <- rep(c("$w=1$", "$w=2$", "$w=3$"), 2)
rownames(arl_sqma11_sacf_sop_bp_tmp) <- c("$10, 10$", "$15,15$", "$25,25$", "$40,25$")
print(xtable(arl_sqma11_sacf_sop_bp_tmp, digits=2, caption = "SOP-BP-Results and SACF-BP-Results for SQMA(1, 1)."),
      type = "latex",
      add.rownames = TRUE,
      include.colnames = TRUE,
      align = "l|lllllllll",
      caption.placement = "top",
      sanitize.rownames.function = function(x) {x},
      sanitize.colnames.function = function(x) {x}   
      )
"""


# ----------------------------------------------------------------------#
# ---------------------      SQMA(2, 2)            ---------------------# 
# ----------------------------------------------------------------------#

# Delays without outliers
arl_sop_sqma22 = load_object("sqma22/arl_sop_sqma22.jld2")
arl_sacf_sqma22 = load_object("sqma22/arl_sacf_sqma22.jld2")
arl_sqma22_sacf_sop = hcat(arl_sop_sqma22, arl_sacf_sqma22)
@rput arl_sqma22_sacf_sop

R"""
library(xtable)
arl_sqma22_sacf_sop_tmp <- round(arl_sqma22_sacf_sop, 3)
colnames(arl_sqma22_sacf_sop_tmp) <- rep(c("$d_1, d_2=1$", "$d_1, d_2=2$", "$d_1, d_2=3$"), 2)
rownames(arl_sqma22_sacf_sop_tmp) <- c("$10, 10$", "$15,15$", "$25,25$", "$40,25$")
print(xtable(arl_sqma22_sacf_sop_tmp, digits=2, caption = "SOP-Results and SACF-Results for SQMA(2, 2)."),
      type = "latex",
      add.rownames = TRUE,
      include.colnames = TRUE,
      align = "l|lllllllll",
      caption.placement = "top",
      sanitize.rownames.function = function(x) {x},
      sanitize.colnames.function = function(x) {x}   
      )
"""

# BP without outliers
arl_sop_sqma22_bp = load_object("sqma22/arl_sop_bp_sqma22.jld2")
arl_sacf_sqma22_bp = load_object("sqma22/arl_sacf_bp_sqma22.jld2")
arl_sqma22_sacf_sop_bp = hcat(arl_sop_sqma22_bp, arl_sacf_sqma22_bp)
@rput arl_sqma22_sacf_sop_bp

R"""
library(xtable)
arl_sqma22_sacf_sop_bp_tmp <- round(arl_sqma22_sacf_sop_bp, 3)
colnames(arl_sqma22_sacf_sop_bp_tmp) <- rep(c("$w=1$", "$w=2$", "$w=3$"), 2)
rownames(arl_sqma22_sacf_sop_bp_tmp) <- c("$10, 10$", "$15,15$", "$25,25$", "$40,25$")
print(xtable(arl_sqma22_sacf_sop_bp_tmp, digits=2, caption = "SOP-BP-Results and SACF-BP-Results for SQMA(2, 2)."),
      type = "latex",
      add.rownames = TRUE,
      include.colnames = TRUE,
      align = "l|lllllllll",
      caption.placement = "top",
      sanitize.rownames.function = function(x) {x},
      sanitize.colnames.function = function(x) {x}   
      )
"""

# ----------------------------------------------------------------------#
# ---------------------       SAR (1)              ---------------------# 
# ----------------------------------------------------------------------#

# Delays without outliers
arl_sop_sar1 = load_object("sar1/arl_sop_sar1.jld2")
arl_sacf_sar1 = load_object("sar1/arl_sacf_sar1.jld2")
arl_sar1_sacf_sop = hcat(arl_sop_sar1, arl_sacf_sar1)
@rput arl_sar1_sacf_sop

R"""
library(xtable)
arl_sar1_sacf_sop_tmp <- round(arl_sar1_sacf_sop, 3)
colnames(arl_sar1_sacf_sop_tmp) <- rep(c("$d_1, d_2=1$", "$d_1, d_2=2$", "$d_1, d_2=3$"), 2)
rownames(arl_sar1_sacf_sop_tmp) <- c("$10, 10$", "$15,15$", "$25,25$", "$40,25$")
print(xtable(arl_sar1_sacf_sop_tmp, digits=2, caption = "SOP-Results and SACF-Results for SAR(1)."),
      type = "latex",
      add.rownames = TRUE,
      include.colnames = TRUE,
      align = "l|lllllllll",
      caption.placement = "top",
      sanitize.rownames.function = function(x) {x},
      sanitize.colnames.function = function(x) {x}   
      )
"""


# Delays with outliers
arl_sop_sar1_outl = load_object("sar1/arl_sop_sar1_outl.jld2")
arl_sacf_sar1_outl = load_object("sar1/arl_sacf_sar1_outl.jld2")
arl_sar1_sacf_sop_outl = hcat(arl_sop_sar1_outl, arl_sacf_sar1_outl)
@rput arl_sar1_sacf_sop_outl

R"""
library(xtable)
arl_sar1_sacf_sop_outl_tmp <- round(arl_sar1_sacf_sop_outl, 3)
colnames(arl_sar1_sacf_sop_outl_tmp) <- rep(c("$d_1, d_2=1$", "$d_1, d_2=2$", "$d_1, d_2=3$"), 2)
rownames(arl_sar1_sacf_sop_outl_tmp) <- c("$10, 10$", "$15,15$", "$25,25$", "$40,25$")
print(xtable(arl_sar1_sacf_sop_outl_tmp, digits=2, caption = "SOP-Results and SACF-Results for SAR(1) with outliers."),
      type = "latex",
      add.rownames = TRUE,
      include.colnames = TRUE,
      align = "l|lllllllll",
      caption.placement = "top",
      sanitize.rownames.function = function(x) {x},
      sanitize.colnames.function = function(x) {x}   
      )
"""

# BP without outliers
arl_sop_sar1_bp = load_object("sar1/arl_sop_bp_sar1.jld2")
arl_sacf_sar1_bp = load_object("sar1/arl_sacf_bp_sar1.jld2")
arl_sar1_sacf_sop_bp = hcat(arl_sop_sar1_bp, arl_sacf_sar1_bp)
@rput arl_sar1_sacf_sop_bp

R"""
library(xtable)
arl_sar1_sacf_sop_bp_tmp <- round(arl_sar1_sacf_sop_bp, 3)
colnames(arl_sar1_sacf_sop_bp_tmp) <- rep(c("$w=1$", "$w=2$", "$w=3$"), 2)
rownames(arl_sar1_sacf_sop_bp_tmp) <- c("$10, 10$", "$15,15$", "$25,25$", "$40,25$")
print(xtable(arl_sar1_sacf_sop_bp_tmp, digits=2, caption = "SOP-BP-Results and SACF-BP-Results for SAR(1)."),
      type = "latex",
      add.rownames = TRUE,
      include.colnames = TRUE,
      align = "l|lllllllll",
      caption.placement = "top",
      sanitize.rownames.function = function(x) {x},
      sanitize.colnames.function = function(x) {x}   
      )
"""

# BP with outliers
arl_sop_sar1_bp_outl = load_object("sar1/arl_sop_bp_sar1_outl.jld2")
arl_sacf_sar1_bp_outl = load_object("sar1/arl_sacf_bp_sar1_outl.jld2")
arl_sar1_sacf_sop_bp_outl = hcat(arl_sop_sar1_bp_outl, arl_sacf_sar1_bp_outl)
@rput arl_sar1_sacf_sop_bp_outl

R"""
library(xtable)
arl_sar1_sacf_sop_bp_outl_tmp <- round(arl_sar1_sacf_sop_bp_outl, 3)
colnames(arl_sar1_sacf_sop_bp_outl_tmp) <- rep(c("$w=1$", "$w=2$", "$w=3$"), 2)
rownames(arl_sar1_sacf_sop_bp_outl_tmp) <- c("$10, 10$", "$15,15$", "$25,25$", "$40,25$")
print(xtable(arl_sar1_sacf_sop_bp_outl_tmp, digits=2, caption = "SOP-BP-Results and SACF-BP-Results for SAR(1) with outliers."),
      type = "latex",
      add.rownames = TRUE,
      include.colnames = TRUE,
      align = "l|lllllllll",
      caption.placement = "top",
      sanitize.rownames.function = function(x) {x},
      sanitize.colnames.function = function(x) {x}   
      )
"""
