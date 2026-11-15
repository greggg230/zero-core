"""zero.farm_def (zero-farm PROTOCOL.md 7, docs/az.md 44.3): definition/1 documents (their checks), the version
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
    # (cards named with the Oracle separator " // " in a game spec are not paths: the Delver 75's Rough // Tumble)
    assert not FD.looks_like_path('{"decks":{"p0":["Rough // Tumble","Tamiyo, Inquisitive Student // Tamiyo, Seasoned Scholar"]}}')
    assert FD.looks_like_path('a //b') and FD.looks_like_path('x // ../y') and FD.looks_like_path('a/b')


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


def _manifests(*mans):
    """{build_id: manifest bytes} (a build id is sha256[:24] of its manifest's bytes) and the ids in order."""
    import hashlib
    raws = [json.dumps(m).encode() for m in mans]
    ids = [hashlib.sha256(r).hexdigest()[:24] for r in raws]
    return dict(zip(ids, raws)), ids


def test_run_alternates_from_their_manifests():
    """A run's alternates (zero-farm PROTOCOL rev 11, 9.4): each a worker build of the default build's game that
    requires a runtime pack; `when.vendor` the pack's `gpu.vendor`, else the caller's vendor (a pack without `gpu`);
    the definition's builds the default then the alternates'. Refused: a vendor that differs from the pack's, a pack
    without one and no vendor, an alternate without `requires_runtime`, of another game, not a worker, requiring a
    build that is not a runtime pack, the default build again, a duplicate, more than 4."""
    cpu = {'manifest': 1, 'kind': 'runtime', 'version': 'ort-1.31.1-cpu', 'files': []}
    cuda = {'manifest': 1, 'kind': 'runtime', 'version': 'cuda-12.8-cudnn-9.28-ort-1.24.4', 'min_agent': '0.2.10',
            'gpu': {'vendor': 'nvidia', 'min_driver': '527.41', 'providers': ['cuda']}, 'files': []}
    winml = {'manifest': 1, 'kind': 'runtime', 'version': 'winml', 'files': []}
    packs, (rt_cpu, rt_cuda, rt_winml) = _manifests(cpu, cuda, winml)
    w = {'manifest': 1, 'kind': 'worker', 'version': 'mtg-1', 'game': 'mtg', 'entry': 'mtgz-worker.exe', 'files': []}
    builds, (b_cpu, b_cuda, b_winml, b_none, b_other, b_rt, b_badpack) = _manifests(
        dict(w, requires_runtime=rt_cpu), dict(w, requires_runtime=rt_cuda), dict(w, requires_runtime=rt_winml),
        w, dict(w, game='zero-toy', requires_runtime=rt_cuda), dict(cuda, version='x'),
        dict(w, requires_runtime='ab' * 12, version='mtg-2'))
    objs = {**packs, **builds, 'ab' * 12: json.dumps(w).encode()}    # ('ab' * 12, b_badpack's pack: not that build's)
    seen = []

    def t(method, url, headers, data):
        bid = url.split('/builds/')[1].split('/')[0]
        seen.append(bid)
        return (200, objs[bid]) if bid in objs else (404, b'{"error": "not_found"}')
    cl = FD.Client('http://fake', token=TOKEN, transport=t)
    # none: the default build alone, nothing fetched
    assert FD.run_alternates(cl, b_cpu, rt_cpu, []) == ([], [b_cpu]) and FD.run_alternates(cl, b_cpu, rt_cpu, None) == ([], [b_cpu])
    assert seen == []
    # the CUDA pack names its vendor; an explicit one must agree
    want = [{'build_id': b_cuda, 'runtime': rt_cuda, 'when': {'vendor': 'nvidia'}}]
    assert FD.run_alternates(cl, b_cpu, rt_cpu, [b_cuda]) == (want, [b_cpu, b_cuda])
    assert FD.run_alternates(cl, b_cpu, rt_cpu, [b_cuda], 'nvidia') == (want, [b_cpu, b_cuda])
    assert seen[:3] == [b_cpu, b_cuda, rt_cuda]
    # a pack without gpu: the caller's vendor; two alternates in order of preference
    alts, bs = FD.run_alternates(cl, b_cpu, rt_cpu, [b_winml], 'amd')
    assert alts == [{'build_id': b_winml, 'runtime': rt_winml, 'when': {'vendor': 'amd'}}] and bs == [b_cpu, b_winml]
    # (an alternate may require the default's pack - only the same (build, runtime) is the default - given a vendor)
    assert FD.run_alternates(cl, b_winml, rt_winml, [b_cpu], 'intel')[0][0]['runtime'] == rt_cpu
    # the definition takes the builds (its format unchanged; its runtime stays the default pack)
    d = FD.definition('run', 'r', 1, 'mtg', FD.run_alternates(cl, b_cpu, rt_cpu, [b_cuda])[1], rt_cpu,
                      {'1': _ck(), '2': _ck()}, {'eval': _eval()}, created=1000)
    assert d['builds'] == [b_cpu, b_cuda] and d['runtime'] == rt_cpu
    assert not FD.same_content(d, dict(d, builds=[b_cpu])), 'a change of builds is a change of content (a new version)'
    for alt, vendor, why in (
            ([b_cuda], 'amd', 'is for nvidia GPUs, not amd'),          # the pack's own vendor differs
            ([b_winml], None, 'names no GPU vendor'),                    # a pack without gpu, no vendor given
            ([b_cuda, b_winml], 'amd', 'is for nvidia GPUs'),            # (one vendor for every alternate)
            ([b_none], None, 'requires no runtime pack'),                # requires_runtime missing
            ([b_other], None, "a worker of 'zero-toy'"),                 # another game
            ([b_rt], None, 'not a worker'),                              # a runtime build as the alternate
            ([b_badpack], None, 'not that build'),                       # its pack's manifest is not that build's
            ([b_cpu], None, "the run's own build"),
            ([b_cuda, b_cuda], None, 'listed twice'),
            ([b_cuda] * 5, None, 'at most 4'),
            ([b_cuda], 'NVIDIA', 'one of'),                              # not a profile vendor name
            (['nope'], None, 'a build_id')):
        with pytest.raises(SystemExit, match=why):
            FD.run_alternates(cl, b_cpu, rt_cpu, alt, vendor)
    # a requires_runtime naming a worker build, not a pack
    builds2, (b_wpack,) = _manifests(dict(w, requires_runtime=b_cpu, version='mtg-3'))
    objs.update(builds2)
    with pytest.raises(SystemExit, match="a 'worker' build, not a runtime pack"):
        FD.run_alternates(cl, b_cpu, rt_cpu, [b_wpack])
    # worker_game keeps refusing a default build whose pack is not the run's runtime
    with pytest.raises(SystemExit, match='requires runtime'):
        FD.worker_game(cl, b_cuda, rt_cpu)


def test_checkpoint_entries(tmp_path):
    m, r = tmp_path / 'model.onnx', tmp_path / 'reference.npz'
    m.write_bytes(b'graph' * 10)
    r.write_bytes(b'rows')
    e = FD.checkpoint_entry(str(m), str(r))
    assert e == {'model_sha256': FD.file_sha256(str(m)), 'model_size': 50, 'reference_sha256': FD.file_sha256(str(r)),
                 'reference_size': 4}
    assert FD.checkpoint_entry(str(m))['reference_sha256'] is None
