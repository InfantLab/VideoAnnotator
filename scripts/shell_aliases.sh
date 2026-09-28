# Optional shell shorthands for VideoAnnotator. Source from ~/.bashrc or
# ~/.zshrc (the dev container does this for you):
#
#   source /path/to/VideoAnnotator/scripts/shell_aliases.sh
#
#   va-start   = scripts/start_server.sh   (sync, set up DB, start server, print viewer link)
#   va         = uv run videoannotator     (the CLI: va job submit ..., va pipelines list)
#
# Docs always spell out the full commands; these are only shorthand for them.
# Run `alias va va-start` to see exactly what each expands to.

_va_root="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/.." && pwd)"

# --project: work from any directory, not just the repo root.
alias va="uv run --project '$_va_root' videoannotator"
alias va-start="'$_va_root/scripts/start_server.sh'"

unset _va_root
