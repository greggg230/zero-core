# Source this (`. tools/local_env.sh`) to read the machine's own settings: the untracked file
# ~/.config/zero/local.env ($ZERO_LOCAL_CONFIG for another), shell-style KEY=value lines, shared by
# zero-core and every game repo (python/zero/local_config.py has the same rules; the keys are in
# tools/local.env.example). A variable already set in the environment wins over the file. Nothing
# in the file is executed: each line is parsed, never sourced.
_zero_local_file="${ZERO_LOCAL_CONFIG:-$HOME/.config/zero/local.env}"
if [ -f "$_zero_local_file" ]; then
  while IFS= read -r _zero_line || [ -n "$_zero_line" ]; do
    _zero_line="${_zero_line%$'\r'}"
    _zero_line="${_zero_line#"${_zero_line%%[![:space:]]*}"}"
    case "$_zero_line" in ''|'#'*) continue ;; esac
    _zero_line="${_zero_line#export }"
    _zero_key="${_zero_line%%=*}"
    [ "$_zero_key" != "$_zero_line" ] || continue
    _zero_key="${_zero_key%"${_zero_key##*[![:space:]]}"}"
    case "$_zero_key" in ''|[0-9]*|*[!A-Za-z0-9_]*) continue ;; esac
    _zero_val="${_zero_line#*=}"
    _zero_val="${_zero_val#"${_zero_val%%[![:space:]]*}"}"
    _zero_val="${_zero_val%"${_zero_val##*[![:space:]]}"}"
    case "$_zero_val" in
      \"*\") _zero_val="${_zero_val#\"}"; _zero_val="${_zero_val%\"}" ;;
      \'*\') _zero_val="${_zero_val#\'}"; _zero_val="${_zero_val%\'}" ;;
    esac
    if [ -z "${!_zero_key:-}" ]; then
      export "$_zero_key=$_zero_val"
    fi
  done < "$_zero_local_file"
fi
unset _zero_local_file _zero_line _zero_key _zero_val
