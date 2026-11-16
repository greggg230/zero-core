"""The machine's own settings, kept out of the repositories (AGENTS.md "Private details").

zero-core and the games are public, so nothing tied to one person's machines is committed: no
ssh targets, host names, IP addresses, home-directory paths or farm machines. A machine that
needs such a setting keeps it in one untracked file, shared by zero-core and every game repo:

    ~/.config/zero/local.env        (another file: $ZERO_LOCAL_CONFIG)

Shell-style `KEY=value` lines; `#` starts a comment line; a value may be quoted ('...' or "...").
`tools/local.env.example` lists the keys. An environment variable of the same name wins over the
file, so a one-off `KEY=value command` still works. Every reader has a generic default for an
unset key (usually "no such machine": the feature is off), so a fresh clone runs as it is.

The same file is read by bash scripts through `tools/local_env.sh` (same rules).

Light on purpose: standard library only, safe to import from anywhere (the run-control tools,
gate scripts, tests).
"""
import os

ENV_FILE = 'ZERO_LOCAL_CONFIG'
DEFAULT_PATH = os.path.join('~', '.config', 'zero', 'local.env')

_cache = {}


def path():
    """-> the local config file's path (it need not exist)."""
    return os.path.expanduser(os.environ.get(ENV_FILE) or DEFAULT_PATH)


def parse(text):
    """-> {key: value} of a local.env text (later lines win)."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[len('export '):].lstrip()
        key, sep, value = line.partition('=')
        key = key.strip()
        if not sep or not key.replace('_', 'a').isalnum() or key[0].isdigit():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
            value = value[1:-1]
        out[key] = value
    return out


def load(refresh=False):
    """-> {key: value} of the local config file ({} when there is none). Cached per path."""
    p = path()
    if refresh or p not in _cache:
        try:
            with open(p, encoding='utf-8') as f:
                _cache[p] = parse(f.read())
        except OSError:
            _cache[p] = {}
    return _cache[p]


def get(key, default=None):
    """-> the setting `key`: the environment's value if set and non-empty, else the local config
    file's, else `default`."""
    v = os.environ.get(key)
    if v:
        return v
    v = load().get(key)
    return v if v else default


def big_disk_path(*parts, default=None):
    """-> a path under the machine's big scratch disk (ZERO_BIG_DISK, e.g. `E:` on a Windows
    desktop whose C: is small), or `default` when the machine names none."""
    root = get('ZERO_BIG_DISK')
    if not root:
        return default
    return '/'.join([root.rstrip('/\\')] + [p.strip('/\\') for p in parts if p])


def portable_path(p):
    """-> `p` with the home directory written as `~` (for paths recorded in committed files:
    a checkpoint's provenance, a report), so no user name lands in a repository."""
    if not p:
        return p
    home = os.path.expanduser('~')
    norm, h = p.replace('\\', '/'), home.replace('\\', '/').rstrip('/')
    if h and (norm == h or norm.lower().startswith(h.lower() + '/')):
        return '~' + norm[len(h):]
    return p
