# Optional shell shorthands for VideoAnnotator. Source from ~/.bashrc or
# ~/.zshrc (the dev container does this for you):
#
#   source /path/to/VideoAnnotator/scripts/shell_aliases.sh
#
#   va-start   = scripts/start_server.sh   (sync, set up DB, start server, print viewer link;
#                                           offers the running server if there is one)
#   va-status  = scripts/start_server.sh --status   (is it running? where?)
#   va-stop    = scripts/start_server.sh --stop     (stop it)
#   va         = uv run videoannotator     (the CLI: va job submit ..., va pipelines list)
#
# Docs always spell out the full commands; these are only shorthand for them.
# Run `alias va va-start va-status va-stop` to see exactly what each expands to.

_va_root="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/.." && pwd)"

# --project: work from any directory, not just the repo root.
alias va="uv run --project '$_va_root' videoannotator"
alias va-start="'$_va_root/scripts/start_server.sh'"
alias va-status="'$_va_root/scripts/start_server.sh' --status"
alias va-stop="'$_va_root/scripts/start_server.sh' --stop"

unset _va_root
