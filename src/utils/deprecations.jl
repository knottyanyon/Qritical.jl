# Deprecation shims — remove before 1.0.0 (see planning/active/ROADMAP.md §5).
#
# `@deprecate_binding old new false` keeps the old name resolvable as `Qritical.Quench`
# (with a deprecation warning) while the trailing `false` suppresses its export, so
# `names(Qritical)` no longer lists it. New code should use `Evolution`/`EvolutionResult`.

# `Base.@deprecate_binding OldName NewName export_flag` is a Julia macro that:
#   1. Creates an alias: `OldName` still resolves to `NewName` (so `Qritical.Quench` → Evolution)
#   2. Issues a deprecation warning when OldName is used: "Quench is deprecated, use Evolution"
#   3. The `false` argument suppresses re-export: OldName won't appear in `names(Qritical)`
# In Python there's no direct equivalent; closest would be: `Quench = Evolution; import warnings; ...`
Base.@deprecate_binding Quench Evolution false   # Quench → Evolution rename (v0.7.0); `false` = do NOT export `Quench`; accessing `Qritical.Quench` still works but prints a deprecation warning
Base.@deprecate_binding QuenchResult EvolutionResult false   # QuenchResult → EvolutionResult; same pattern; both suppress export so `names(Qritical)` is clean
