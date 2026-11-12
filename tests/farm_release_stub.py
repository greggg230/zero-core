"""A stand-in for zero-farm's `farm-release define` (zero-farm PROTOCOL.md 7) in tests: it records its arguments,
checks the admin token arrived in its environment (never in its arguments), writes a fake signature next to the
definition and fakes the coordinator's registration in a JSON state file the tests' fake coordinator reads.

  python farm_release_stub.py define <definition.json> --key <data key> [--publish --url URL]

Environment: FARM_STUB_LOG (JSON lines: argv, whether the token was in the environment and equal to
FARM_STUB_TOKEN), FARM_STUB_STATE (the fake coordinator's definitions: {"<kind>/<id>": {"version", "sha256",
"key", "doc"}}), FARM_STUB_FAIL (exit 1 with that message), FARM_STUB_ECHO_TOKEN (print the token: the caller
must scrub it)."""
import hashlib
import json
import os
import sys


def main(argv):
    pos, flags = [], {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith('--'):
            if i + 1 < len(argv) and not argv[i + 1].startswith('--'):
                flags[a[2:]] = argv[i + 1]
                i += 2
                continue
            flags[a[2:]] = None
        else:
            pos.append(a)
        i += 1
    tok = os.environ.get('ZEROFARM_ADMIN_TOKEN')
    log = os.environ.get('FARM_STUB_LOG')
    if log:
        with open(log, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'argv': argv, 'token_env': tok is not None,
                                'token_ok': tok is not None and tok == os.environ.get('FARM_STUB_TOKEN')}) + '\n')
    if not pos or pos[0] != 'define' or len(pos) != 2:
        print('usage: define <definition.json> --key K [--publish --url U]', file=sys.stderr)
        return 2
    if 'token' in flags:
        print('stub: the admin token must not be on the command line', file=sys.stderr)
        return 2
    if not flags.get('key') or not os.path.exists(flags['key']):
        print(f'no data key: {flags.get("key")}', file=sys.stderr)
        return 1
    if os.environ.get('FARM_STUB_FAIL'):
        print(os.environ['FARM_STUB_FAIL'], file=sys.stderr)
        return 1
    if os.environ.get('FARM_STUB_ECHO_TOKEN') and tok:
        print(f'debug: token {tok}', file=sys.stderr)
    path = pos[1]
    with open(path, 'rb') as f:
        b = f.read()
    doc = json.loads(b)
    if doc.get('definition') != 1 or doc.get('kind') not in ('run', 'campaign') or not isinstance(doc.get('version'), int):
        print('not a definition/1 document', file=sys.stderr)
        return 1
    sha = hashlib.sha256(b).hexdigest()
    with open(path + '.minisig', 'w', encoding='utf-8', newline='\n') as f:
        f.write(f'untrusted comment: stub signature\nSTUB\ntrusted comment: kind={doc["kind"]} id={doc["id"]} '
                f'version={doc["version"]} sha256={sha}\nSTUB\n')
    key = f'defs/{doc["kind"]}/{doc["id"]}/{doc["version"]}/definition.json'
    print(f'signed {path} (version {doc["version"]}, sha256 {sha})')
    if 'publish' in flags:
        if not flags.get('url') or not tok:
            print('--publish needs --url and ZEROFARM_ADMIN_TOKEN', file=sys.stderr)
            return 1
        sp = os.environ.get('FARM_STUB_STATE')
        st = {}
        if sp and os.path.exists(sp):
            with open(sp, encoding='utf-8') as f:
                st = json.load(f)
        k = f'{doc["kind"]}/{doc["id"]}'
        cur = st.get(k)
        if cur and cur['version'] >= doc['version']:
            print(f'409 conflict: version {doc["version"]} is not above the current {cur["version"]}', file=sys.stderr)
            return 1
        st[k] = {'version': doc['version'], 'sha256': sha, 'key': key, 'doc': doc}
        if sp:
            with open(sp, 'w', encoding='utf-8') as f:
                json.dump(st, f)
        print(f'registered {k} version {doc["version"]} at {flags["url"]} ({key})')
    return 0


def setup(monkeypatch, tmp_path, token):
    """The tests' environment for the stub -> {"exe", "key", "state", "log"}: farm-release = this stub, a data key
    file, the fake coordinator's definitions, the log of calls."""
    d = tmp_path / 'farm-release-stub'
    d.mkdir(exist_ok=True)
    key = d / 'data.key'
    key.write_text('a stand-in data key', encoding='utf-8')
    out = {'exe': os.path.abspath(__file__), 'key': str(key), 'state': str(d / 'coordinator.json'), 'log': str(d / 'calls.jsonl')}
    monkeypatch.setenv('FARM_STUB_STATE', out['state'])
    monkeypatch.setenv('FARM_STUB_LOG', out['log'])
    monkeypatch.setenv('FARM_STUB_TOKEN', token)
    monkeypatch.setenv('ZEROFARM_FARM_RELEASE', out['exe'])
    monkeypatch.setenv('ZEROFARM_DATA_KEY', out['key'])
    monkeypatch.setenv('ZEROFARM_DEFS_DIR', str(tmp_path / 'definitions'))
    return out


def calls(log):
    if not os.path.exists(log):
        return []
    with open(log, encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def current(state, kind, id):
    """The fake coordinator's current definition of (kind, id) ({version, sha256, key, doc}) or None."""
    if not state or not os.path.exists(state):
        return None
    with open(state, encoding='utf-8') as f:
        return json.load(f).get(f'{kind}/{id}')


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
