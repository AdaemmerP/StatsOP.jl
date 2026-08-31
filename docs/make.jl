using StatsOrdinalPatterns
using QuartoDocBuilder

# QuartoDocBuilder writes to "docs/reference", relative to the working directory,
# so the script has to run from the repository root.
cd(dirname(@__DIR__))

# Regenerate the reference pages from the docstrings only. `quarto_build_site`
# would additionally rewrite `docs/_quarto.yml` with `force=true` and thereby
# discard the hand-curated navbar and sidebar.
quarto_rebuild_reference(StatsOrdinalPatterns)
