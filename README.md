<a ><img src='docs/op_logo.svg' align="left" height="80" /></a>

#  StatsOP.jl 

[![CI](https://github.com/AdaemmerP/StatsOP.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/AdaemmerP/StatsOP.jl/actions/workflows/CI.yml)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://adaemmerp.github.io/StatsOP.jl/)
[![Coverage](https://codecov.io/gh/AdaemmerP/StatsOP.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/AdaemmerP/StatsOP.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

StatsOP.jl provides (sequential) tests and control charts for time series and spatial data based on ordinal patterns, building on the seminal work of Bandt and Pompe (2002). It covers:

- **Ordinal patterns (OP)** — tests, control charts, and dependence/changepoint detection for univariate time series.
- **Generalized ordinal patterns (GOP)** — extends OP to account for ties, for discrete-valued (count) series.
- **Spatial ordinal patterns (SOP)** — the 2D analogue of OP, for image-like / spatial lattice data.
- **Spatial autocorrelation (SACF)** — complementary tools for spatial and qualitative process monitoring.

For most tests, three variants are available: an asymptotic test, a bootstrap test, and (for OP) a surrogate-data test — see the [documentation](https://adaemmerp.github.io/StatsOP.jl/) for details and worked examples.

Previously, the name of the package was **OrdinalPatterns.jl**, but it has been renamed to better reflect its purpose.

## Installation

Once registered, install the latest release from the General registry:

```julia
using Pkg
Pkg.add("StatsOP")
```

Until then, install the development version directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/AdaemmerP/StatsOP.jl")
```

## Authors

Philipp Adämmer, Philipp Wittenberg, Christian Weiß

