"""zero.farm_def (zero-farm PROTOCOL.md 7, docs/az.md 42.3): definition/1 documents (their checks), the version
(the coordinator's current or the archive's newest, + 1), the archive (written once, never other bytes under a
version), signing and publishing through farm-release (a stub: tests/farm_release_stub.py) with the admin token in
its environment only, and the check that the coordinator names the published version."""
import json
import os

import pytest

from zero import farm_def as FD
import farm_release_stub as STUB

TOKEN = 'secret-admin-token-def'
B = 'ab' * 12
R = 'cd' * 12
SHA = 'ef' * 32


def _ck():
    return {'model_sha256': SHA, 'model_size': 100, 'reference_sha256': SHA, 'reference_size': 10}


def _eval():
    return {'settings': {'game': 'toy', 'eval_sims': 4}, 'seats': [{'net': 1}, {'net': 2}], 'paired': True}


class Fake:
    """GET /v1/admin/run|campaign/{id} with the stub's registered definition (or none, or a fixed answer)."""

    def __init__(self, answer=None, exists=True):
        self.answer, self.exists, self.calls = answer, exists, []

    def __call__(self, method, url, headers, data):
        assert headers['authorization'] == f'Bearer {TOKEN}'
        path = url.split('://', 1)[1].split('/', 1)[1]
        self.calls.append((method, '/' + path))
        kind, id = path.split('/')[2:4]
        if not self.exists:
            return 404, json.dumps({'error': 'not_found', 'message': id}).encode()
        if self.answer is not None:
            return 200, json.dumps(self.answer).encode()
        cur = STUB.current(os.environ.get('FARM_STUB_STATE'), kind, id)
        return 200, json.dumps({kind: id, 'definition': cur and {k: cur[k] for k in ('version', 'sha256', 'key')}}).encode()


def test_documents_and_their_checks():
    d = FD.definition('run', 'h2h-x', 3, 'toy', [B], R, {'2': _ck(), '1': _ck()}, {'eval': _eval()}, created=1000,
                      expires=FD.expires_at(1000, 1))
    assert list(d) == ['definition', 'kind', 'id', 'version', 'created', 'expires', 'game', 'builds', 'runtime',
                       'checkpoints', 'jobs']
    assert list(d['checkpoints']) == ['1', '2'] and d['expires'] == 1000 + 86400
    assert d['jobs'] == {'selfplay': None, 'eval': _eval(), 'fuzz': None}
    assert FD.encode(d) == FD.encode(json.loads(FD.encode(d)))
    c = FD.definition('campaign', 'delver-fuzz-4b44b07', 1, 'mtg', [B], None, None,
                      {'fuzz': {'mode': 'delver-games', 'env': {'FUZZ_POOL': 'delver'}}}, created=5)
    assert c['expires'] is None and c['checkpoints'] == {} and c['runtime'] is None
    assert FD.same_content(c, dict(c, version=2, created=9)) and not FD.same_content(c, dict(c, builds=[R]))
    bad = [
        dict(kind='campaign', jobs={'eval': _eval()}),                                    # a campaign runs fuzz only
        dict(kind='run', jobs={'fuzz': {'mode': 'games', 'env': {}}}, checkpoints={}),    # a run has no fuzz job
        dict(builds=['xyz']),                                                             # a build id
        dict(builds=[B, B]),
        dict(runtime='nope'),
        dict(jobs={'eval': dict(_eval(), games=10)}),                                     # a per-lease field
        dict(jobs={'eval': dict(_eval(), seed=1)}),                                       # (the seed is one too)
        dict(jobs={'eval': dict(_eval(), settings={'resolver_init': 'runs/x/models/iter_0001.pt'})}),   # a path
        dict(jobs={'eval': dict(_eval(), settings={'spec': 'C:decks'})}),                 # a drive
        dict(jobs={'eval': dict(_eval(), settings={'x': ['a', 'b..c']})}),                # '..' in a list
        dict(jobs={'eval': dict(_eval(), settings={'a/b': 1})}),                          # a key
        dict(jobs={'eval': dict(_eval(), seats=[{'net': 1}, {'bot': 'ext:jsbot'}])}),     # an external bot
        dict(jobs={'selfplay': {'settings': {}, 'seats': []}}),                           # not a self-play key
        dict(game='mtg_eoe'),                                                             # a settings game: not a slug
        dict(checkpoints={'1': dict(_ck(), model_size=(1 << 30) + 1), '2': _ck()}),       # over 1 GiB
        dict(jobs={'eval': dict(_eval(), seats=[{'net': 1}, {'net': 3}])}),               # gen 3 is not signed
        dict(jobs={'eval': dict(_eval(), paired='yes')}),
        dict(jobs={}),                                                                    # no job at all
        dict(id='Bad_Id'),
        dict(version=0),
        dict(checkpoints={'1': dict(_ck(), model_sha256='00')}),
        dict(checkpoints={'1': dict(_ck(), reference_size=None)}),
        dict(expires=500),                                                                # before created
    ]
    for over in bad:
        kw = dict(kind='run', id='h2h-x', version=1, game='toy', builds=[B], runtime=R, checkpoints={'1': _ck(), '2': _ck()},
                  jobs={'eval': _eval()}, created=1000, expires=None)
        kw.update(over)
        with pytest.raises(ValueError):
            FD.definition(**kw)
    for fz in ({'mode': 'x-games', 'env': {'A B': '1'}}, {'mode': 'x-games', 'env': {}, 'first': 3},
               {'mode': 'vast:games', 'env': {}}, {'mode': 'x-games', 'env': {'FUZZ_DECK_CHOICE_BY': 'E:/decks.json'}},
               {'mode': 'x-games', 'announce_result': 'yes'}):
        with pytest.raises(ValueError):
            FD.definition('campaign', 'c', 1, 'mtg', [B], jobs={'fuzz': fz})
    FD.definition('campaign', 'c', 1, 'mtg', [B], jobs={'fuzz': {'mode': 'x-games', 'announce_result': False}})
    assert FD.looks_like_path('C:') and FD.looks_like_path('..') and not FD.looks_like_path('cq-n4-v50')
    assert not FD.looks_like_path('{"sealed":{"set":"eoe"},"decks":"eval"}')


def test_versions_the_archive_and_publishing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('ZEROFARM_ADMIN_TOKEN', TOKEN)
    stub = STUB.setup(monkeypatch, tmp_path, TOKEN)
    cl = FD.Client('http://fake', transport=Fake())
    jobs = {'fuzz': {'mode': 'delver-games', 'env': {'FUZZ_POOL': 'delver'}}}
    # a new campaign: version 1; published (token in the environment only), current at the coordinator
    info = FD.prepare(cl, 'campaign', 'cmp', 'mtg', [B], None, None, jobs, expires_days=90, now=2000)
    assert info['version'] == 1 and info['expires'] == 2000 + 90 * 86400 and info['key'] == 'defs/campaign/cmp/1/definition.json'
    assert info['path'] == os.path.join(str(tmp_path / 'definitions'), 'campaign', 'cmp', '1', 'definition.json')
    out = FD.publish(cl, info)
    assert out['published']
    assert FD.define_result('noise\n{"kind": "campaign", "version": 1}\n') == {'kind': 'campaign', 'version': 1}
    assert FD.define_result('no json here') is None
    calls = STUB.calls(stub['log'])
    assert calls[-1]['argv'] == ['define', info['path'], '--key', stub['key'], '--publish', '--url', 'http://fake']
    assert calls[-1]['token_ok'] and TOKEN not in json.dumps(calls)
    assert STUB.current(stub['state'], 'campaign', 'cmp')['version'] == 1
    assert FD.published_to(info['path'], 'http://fake') and not FD.published_to(info['path'], 'http://other')
    # the next one: 2; the archive keeps both, each under its version, never rewritten with other bytes
    info2 = FD.prepare(cl, 'campaign', 'cmp', 'mtg', [R], None, None, jobs, now=2001)
    assert info2['version'] == 2 and FD.local_versions('campaign', 'cmp') == [1, 2]
    doc1, _ = FD.load_local('campaign', 'cmp', 1)
    with pytest.raises(SystemExit):
        FD.write(dict(doc1, builds=[R]))
    assert FD.write(doc1) == info['path']                       # the same bytes again: fine
    FD.publish(cl, info2)
    # the coordinator's version counts when it is ahead of the archive
    cl9 = FD.Client('http://fake', transport=Fake(answer={'definition': {'version': 9, 'sha256': SHA}}))
    assert FD.next_version(cl9, 'campaign', 'cmp') == 10
    assert FD.next_version(FD.Client('http://fake', transport=Fake(exists=False)), 'run', 'new-run') == 1
    # an old coordinator (no definition in its answer): a note, not an error; one that names another version: an error
    info_old = FD.prepare(cl, 'campaign', 'cmp', 'mtg', [B], None, None, jobs, now=2002)
    assert info_old['version'] == 3
    FD.publish(FD.Client('http://fake', transport=Fake(answer={'campaign': 'cmp'})), info_old)
    assert 'names no definition' in capsys.readouterr().err
    info3 = FD.prepare(cl, 'campaign', 'cmp', 'mtg', [B], None, None, jobs, now=2002)
    with pytest.raises(SystemExit, match='not the one just published'):
        FD.publish(FD.Client('http://fake', transport=Fake(answer={'definition': {'version': 2}})), info3)
    # farm-release fails: loudly, with its message, the token scrubbed from what it printed
    monkeypatch.setenv('FARM_STUB_ECHO_TOKEN', '1')
    monkeypatch.setenv('FARM_STUB_FAIL', 'boom: the coordinator refused it')
    info4 = FD.prepare(cl, 'campaign', 'cmp', 'mtg', [B], None, None, jobs, now=2003)
    with pytest.raises(SystemExit) as e:
        FD.publish(cl, info4)
    assert 'boom: the coordinator refused it' in str(e.value) and not FD.published_to(info4['path'], 'http://fake')
    monkeypatch.delenv('FARM_STUB_FAIL')
    FD.publish(cl, info4)
    err = capsys.readouterr().err
    assert TOKEN not in err
    # no farm-release, no key: refused before anything runs
    with pytest.raises(SystemExit, match='farm-release not found'):
        FD.publish(cl, info4, farm_release=str(tmp_path / 'missing.exe'))
    with pytest.raises(SystemExit, match='data key not found'):
        FD.publish(cl, info4, key=str(tmp_path / 'missing.key'))


def test_a_worker_builds_game_from_its_manifest(monkeypatch):
    """A definition's game is the worker build's (its manifest: "mtg" for MTG Zero, not a run's "mtg_eoe"); the
    manifest must be the build's (sha256[:24] = its id), a worker's, and require the runtime given."""
    import hashlib
    man = {'manifest': 1, 'kind': 'worker', 'game': 'mtg', 'requires_runtime': R}
    raw = json.dumps(man).encode()
    bid = hashlib.sha256(raw).hexdigest()[:24]
    other = json.dumps(dict(man, kind='runtime')).encode()
    rid = hashlib.sha256(other).hexdigest()[:24]
    objs = {bid: raw, rid: other, B: raw}                     # (B: an id whose manifest is another build's)

    def t(method, url, headers, data):
        return 200, objs[url.split('/builds/')[1].split('/')[0]]
    cl = FD.Client('http://fake', token=TOKEN, transport=t)
    assert FD.worker_game(cl, bid, R) == 'mtg'
    for b, rt, why in ((bid, B, 'requires runtime'), (rid, R, 'not a worker'), (B, R, 'not that build'),
                       ('nope', R, 'a build_id')):
        with pytest.raises(SystemExit, match=why):
            FD.worker_game(cl, b, rt)


def test_checkpoint_entries(tmp_path):
    m, r = tmp_path / 'model.onnx', tmp_path / 'reference.npz'
    m.write_bytes(b'graph' * 10)
    r.write_bytes(b'rows')
    e = FD.checkpoint_entry(str(m), str(r))
    assert e == {'model_sha256': FD.file_sha256(str(m)), 'model_size': 50, 'reference_sha256': FD.file_sha256(str(r)),
                 'reference_size': 4}
    assert FD.checkpoint_entry(str(m))['reference_sha256'] is None
