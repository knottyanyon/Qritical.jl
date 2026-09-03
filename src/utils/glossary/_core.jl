#=META
source:
  author: Bavithra
  coauthor:
  reviewer:
docstrings:
  author: Bavithra
  coauthor:
  reviewer:
refs:
credits:
=#

using Glossaries

# Installs current_glossary()/current_glossary!() in Qritical, backing the code-level
# glossary used by Glossaries.Argument()/Glossaries.Keyword() docstring interpolation.
# Must run exactly once, before any src/utils/glossary/<theme>.jl file calls @define!.
Glossaries.@Glossary()
