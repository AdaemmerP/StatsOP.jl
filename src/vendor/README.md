# Vendored: GeneralizedChisqDistribution.jl

`GeneralizedChisqDistribution.jl` and `GChisqComputations.jl` in this directory are
**verbatim, unmodified copies** of the upstream sources. They are vendored rather than
taken as a package dependency because the computation method StatsOP relies on lives on
an unregistered branch.

## Provenance

| | |
|---|---|
| Repository | <https://github.com/AdaemmerP/GeneralizedChisqDistribution.jl> |
| Branch     | `revise-computation` |
| Commit     | `665424e920927a4513ac6760807e16738713eb61` |
| Tree       | `842938badf1ca4a49f9c58f369acf5d7766a75c0` |
| Date       | 2026-03-16 |
| Version    | 1.0.1 (upstream `Project.toml`) |
| Author     | Helios De Rosario — see `LICENSE.GeneralizedChisqDistribution` (MIT) |

## How it is wired in

`src/StatsOP.jl` does a single

```julia
include("vendor/GeneralizedChisqDistribution.jl")
```

which defines the submodule `StatsOP.GeneralizedChisqDistribution` (with its own private
inner module `GChisqComputations`). Nothing from it is re-exported by StatsOP.

Call sites reach the type through the internal alias defined in `src/StatsOP.jl`:

```julia
const _GChisqDist = GeneralizedChisqDistribution.GeneralizedChisq
```

This keeps the vendored code fully namespaced, so the name `GeneralizedChisq` cannot
collide with anything StatsOP loads — including a future `Distributions.jl` that absorbs
this distribution.

`StatsOP.GeneralizedChisqDistribution.GeneralizedChisq` is a *distinct type* from the one
in the upstream package. Do not accept or return it across StatsOP's public API; it is
only used internally to build null distributions for p-values and critical values.

## Dependencies the vendored code needs

`Distributions`, `Statistics`, `Random`, `QuadGK` — all present in StatsOP's
`Project.toml` (`QuadGK` was added for this vendoring).

## Updating

Re-sync is a plain file copy, since the sources are unmodified:

```bash
git clone --branch revise-computation \
  https://github.com/AdaemmerP/GeneralizedChisqDistribution.jl /tmp/gchisq
cp /tmp/gchisq/src/{GeneralizedChisqDistribution.jl,GChisqComputations.jl} src/vendor/
```

Then update the commit/tree/date in the table above.
