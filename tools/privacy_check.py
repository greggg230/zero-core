#!/usr/bin/env python3
"""Refuse private details in the repository's tracked files (AGENTS.md "Private details").

    python tools/privacy_check.py [--root DIR] [PATH ...]

The repositories are public. This scans every tracked file (`git ls-files`; or only the PATHs
given) for:

* the committed GENERIC patterns below: Tailscale (CGNAT 100.64/10) addresses and MagicDNS names,
  farm machine ids (`m_` + 26 base32 characters), home-directory paths that name a user
  (`C:/Users/<name>`, `/c/Users/<name>`, `/Users/<name>`; placeholders such as `<user>` pass),
  Windows' and macOS's default PC host names (`DESKTOP-` + 7 characters, `<Name>s-MacBook-Pro`), and
  ssh targets written as `user@<IP>`. A line carrying the marker `privacy-check: allow`, or a whole file
  with `privacy-check: allow-file` in its first 5 lines, is spared these generic rules (synthetic data
  that must look real, e.g. a redactor and its tests; never for real names);
* the machine's LOCAL patterns: one regex per line (case-insensitive, `#` comments) in the untracked
  file ~/.config/zero/privacy_patterns.txt ($ZERO_PRIVACY_PATTERNS for another): the real names of
  hosts, farm machines and accounts; they apply to every file, markers or not. They are never
  committed (that would publish them);
  tools/privacy_patterns.example.txt shows the format.

Binary files are skipped. Prints each finding as `path:line: [rule] excerpt`, then one line
`privacy check: N file(s), M finding(s) (...)`; exit 0 when there is none, 1 when there are, 2 when
it cannot run (not a git checkout, a bad local pattern). The same rules live in mtg-zero's and
zero-farm's copies of this check.
"""
import argparse
import os
import re
import subprocess
import sys

USER_PLACEHOLDERS = (r'<[^<>\s/\\]*>|\.\.\.|\*|%USERNAME%|%USERPROFILE%|\$\{?USER(?:NAME)?\}?|USER(?:NAME)?|'
                     r'user|you|me|someone|example|Public|Default|All Users|runner|runneradmin|[A-Za-z]')
ALLOW = 'privacy-check: allow'     # a line carrying this is skipped (synthetic test data of a redactor, ...)
_SEP = r'(?:\\\\|\\|/)'
_END = r'(?=[\\/"\'`\s,;:)\]}]|$)'

GENERIC = (
    ('tailscale-ip', re.compile(r'(?<![\d.])100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}(?![\d.]*\d)')),
    ('tailscale-name', re.compile(r'\b[A-Za-z0-9-]+\.ts\.net\b')),
    ('farm-machine-id', re.compile(r'(?<![0-9A-Za-z])m_(?=[0-9]*[a-z])[0-9a-z]{26}(?![0-9A-Za-z])')),
    ('windows-home-path', re.compile(r'(?:\b[A-Za-z]:|(?<![\w.-])/[A-Za-z])' + _SEP + r'+Users' + _SEP +
                                     r'+(?!(?:' + USER_PLACEHOLDERS + r')' + _END + r')[^\\/"\'`\s<>]', re.I)),
    ('mac-home-path', re.compile(r'(?<![\w.:\\/-])/Users/(?!(?:' + USER_PLACEHOLDERS + r')' + _END +
                                 r')[A-Za-z0-9._-]+')),
    ('default-pc-hostname', re.compile(r'\b(?:DESKTOP-[A-Z0-9]{7}|LAPTOP-[A-Z0-9]{8})\b', re.I)),
    ('default-mac-hostname', re.compile(r'\b[A-Z][a-z]+s-(?:MacBook-(?:Pro|Air)|iMac|Mac-mini|Mac-Studio)\b')),
    ('ssh-user-at-ip', re.compile(r'\b[a-z_][a-z0-9_.-]*@(?!(?:192\.0\.2|198\.51\.100|203\.0\.113|127\.0\.0)\.)'
                                  r'(?:\d{1,3}\.){3}\d{1,3}\b', re.I)),
)

LOCAL_ENV = 'ZERO_PRIVACY_PATTERNS'
LOCAL_DEFAULT = os.path.join('~', '.config', 'zero', 'privacy_patterns.txt')


def local_patterns_path():
    return os.path.expanduser(os.environ.get(LOCAL_ENV) or LOCAL_DEFAULT)


def load_local(path):
    """-> [(rule, regex)] of the local pattern file ([] when there is none); ValueError on a bad regex."""
    try:
        with open(path, encoding='utf-8') as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    out = []
    for n, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            out.append((f'local pattern {n}', re.compile(line, re.I)))
        except re.error as e:
            raise ValueError(f'{path}:{n}: bad regex {line!r}: {e}')
    return out


def tracked_files(root):
    out = subprocess.run(['git', 'ls-files', '-z'], cwd=root, capture_output=True, check=True).stdout
    return [p for p in out.decode('utf-8', 'surrogateescape').split('\0') if p]


def scan_text(text, rules):
    """-> [(line number, rule, excerpt)] of every match in `text`."""
    hits = []
    for rule, rx in rules:
        for m in rx.finditer(text):
            s = text.rfind('\n', 0, m.start()) + 1
            e = text.find('\n', m.end())
            if ALLOW in text[s:len(text) if e < 0 else e]:
                continue
            line = text.count('\n', 0, m.start()) + 1
            a = max(s, m.start() - 60)
            b = text.find('\n', m.end())
            b = min(len(text) if b < 0 else b, m.end() + 60)
            hits.append((line, rule, text[a:b].strip()))
    return sorted(hits)


ALLOW_FILE = 'privacy-check: allow-file'


def scan(root, paths, rules):
    """-> (files scanned, [(path, line, rule, excerpt)])."""
    generic = {id(rx) for _, rx in GENERIC}
    findings, n = [], 0
    for rel in paths:
        full = os.path.join(root, rel)
        if not os.path.isfile(full):            # (a submodule's gitlink, a file deleted in the work tree)
            continue
        with open(full, 'rb') as f:
            data = f.read()
        if b'\0' in data[:8192]:
            continue
        n += 1
        text = data.decode('utf-8', 'replace')
        use = rules
        if ALLOW_FILE in '\n'.join(text.split('\n', 5)[:5]):
            use = [r for r in rules if id(r[1]) not in generic]
        findings.extend((rel.replace('\\', '/'), *h) for h in scan_text(text, use))
    return n, findings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help='the checkout to scan (default: this one)')
    ap.add_argument('paths', nargs='*', help='only these files (relative to --root)')
    args = ap.parse_args(argv)
    lp = local_patterns_path()
    try:
        local = load_local(lp)
        paths = args.paths or tracked_files(args.root)
    except (ValueError, OSError, subprocess.CalledProcessError) as e:
        print(f'privacy check: cannot run: {e}', file=sys.stderr)
        return 2
    n, findings = scan(args.root, paths, list(GENERIC) + local)
    for path, line, rule, excerpt in findings:
        print(f'{path}:{line}: [{rule}] {excerpt[:200]}')
    where = f'{len(local)} local pattern(s) from {lp}' if local else f'no local patterns ({lp} absent or empty)'
    print(f'privacy check: {n} file(s), {len(findings)} finding(s) ({len(GENERIC)} generic rules, {where})')
    return 1 if findings else 0


if __name__ == '__main__':
    sys.exit(main())
