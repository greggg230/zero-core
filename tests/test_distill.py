"""Distillation (zero/distill.py, tools/distill.py; docs/az.md 13) and fork_run --net: a network's
answers on a window are the evaluator's; the agreement of a network with itself is exact; a student of
another size and architecture converges to its teacher on held-out positions; the tool writes a
checkpoint the encoding guard accepts, with its record, and resumes from a partial one; --compare;
shards of another encoding refused; a run continued with the distilled network as its checkpoint K
(no optimizer state, K's Elo results dropped, the size in its config) trains it on."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from zero import data, distill as D, games as G, net
from zero.infer import NetEvaluator, UniformEvaluator, drive

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYPATH = os.pathsep.join([os.path.join(ROOT, 'python')] + [p for p in [os.environ.get('PYTHONPATH')] if p])


@pytest.fixture(scope='module')
def toy_data(tmp_path_factory):
    """Toy self-play shards (uniform search, every decision recorded): 3 training shards and a
    held-out one in another directory."""
    root = tmp_path_factory.mktemp('distill')
    info = G.info('toy')
    meta = {'game': 'toy', 'features': info['num_features'], 'actions': info['num_actions']}
    for name, games, seed in (('iter_0001', 120, 3), ('iter_0002', 40, 4)):
        seat = G.nn_seat('m', 8, selfplay=True, sims_fast=4, p_full=1.0)
        drv = G.Driver('toy', [[seat, seat]] * games, concurrency=48, threads=2, seed=seed, record=True)
        w = data.ShardWriter(str(root / 'data' / name), 'c00', rows=1200, meta=meta)
        drive(drv, {'m': UniformEvaluator()}, on_batch=lambda lb: w.add(drv.take_samples()))
        w.add(drv.take_samples())
        w.close()
    return root


def teacher(seed=5):
    """A toy MLP with a non-trivial policy and value (random init)."""
    torch.manual_seed(seed)
    return net.make_net(G.info('toy'), 'toy', 'mlp', 64, 2)


def windows(root):
    F, A = G.info('toy')['num_features'], G.info('toy')['num_actions']
    return (data.Window(data.shard_paths(str(root / 'data' / 'iter_0001')), F, A),
            data.Window(data.shard_paths(str(root / 'data' / 'iter_0002')), F, A))


def test_outputs_are_the_evaluators_answers(toy_data):
    W, _ = windows(toy_data)
    t = teacher()
    T = D.outputs(t, W, batch=97)                   # several batches, a ragged last one
    rows = np.arange(W.N)
    pos, _, lens = D._option_index(W, rows)
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    pri, val, mar = NetEvaluator(t).evaluate(W.features(rows).astype(np.float32), W.n_players.astype(np.int32), off,
                                             W.opt_action[pos].astype(np.int32))
    assert np.allclose(pri, T['pi'][pos], atol=1e-6) and np.allclose(val, T['v'], atol=1e-6)
    assert np.allclose(mar, T['m'], atol=1e-5)


def test_agreement_with_itself_is_exact(toy_data):
    W, _ = windows(toy_data)
    T = D.outputs(teacher(), W)
    a = D.agreement(T, T, W)
    assert a['policy_kl'] == 0 and a['top1'] == 1.0 and a['value_mse'] == 0 and a['margin_mse'] == 0
    assert a['net'] == a['teacher'] and a['decisions'] > 0 and a['rows'] == W.N
    b = D.agreement(D.outputs(teacher(6), W), T, W)
    assert b['policy_kl'] > 0.01 and b['top1'] < 0.9 and b['value_mse'] > 0


@pytest.mark.parametrize('arch,hidden,blocks', [('tokens', 32, 2), ('mlp', 128, 3)])
def test_a_student_of_another_size_and_arch_converges_to_its_teacher(toy_data, arch, hidden, blocks):
    W, H = windows(toy_data)
    t = teacher()
    T, TH = D.outputs(t, W), D.outputs(t, H)
    s = D.new_student(t, arch, hidden, blocks, heads=2, seed=1)
    assert s.config.get('arch', 'mlp') == arch
    before = D.agreement(D.outputs(s, H), TH, H)
    reports = []
    D.distill(s, W, T, steps=300, batch=128, lr=2e-3, warmup=30, prefetch=2, every=100,
              on_report=lambda k, terms, lr: reports.append(terms))
    after = D.agreement(D.outputs(s, H), TH, H)
    assert len(reports) == 3 and reports[-1]['kl_t'] < reports[0]['kl_t']
    assert after['policy_kl'] < before['policy_kl'] / 4 and after['policy_kl'] < 0.03
    assert after['top1'] > 0.85 and after['value_mse'] < before['value_mse'] / 5


def test_teacher_weight_zero_is_the_runs_supervised_loss(toy_data):
    """teacher_w 0: the loss is the run's own (policy CE against pi, value CE against the value_q
    blend, margin); teacher_w 1: the same terms against the teacher's answers."""
    import torch.nn.functional as F_
    W, _ = windows(toy_data)
    t = teacher()
    b = D.Taught(W, D.outputs(t, W)).batch(np.arange(64))
    s = D.new_student(t, 'mlp', 32, 1, seed=2).train()
    total, terms = D.losses(s, b, teacher_w=0.0, value_q=0.3, margin_w=0.5)
    plog, vlog, mp = s(b['x'], b['n_players'])
    S = s.seats
    logp = F_.log_softmax(plog.masked_fill(~b['mask'], -1e9), 1)
    lp = -(b['pi'] * logp).sum(1).mean()
    zt = (0.7 * b['z'] + 0.3 * b['q'])[:, :S]
    lv = -(zt * F_.log_softmax(vlog, 1).clamp(min=-1e4)).sum(1).mean()
    present = (torch.arange(S)[None, :] < b['n_players'][:, None]).float()
    lm = (((mp - b['sc'][:, :S]) ** 2) * present).sum(1).mean() / 2
    assert torch.allclose(total, lp + lv + 0.5 * lm, atol=1e-5)
    total1, terms1 = D.losses(s, b, teacher_w=1.0, value_q=0.3, margin_w=0.5)
    assert torch.allclose(total1, terms1['policy_t'] + terms1['value_t'] + 0.5 * terms1['margin_t'], atol=1e-5)
    # the dense teacher policy is the teacher's softmax over the legal actions
    assert torch.allclose(b['tpi'].sum(1), torch.ones(64), atol=1e-5) and not (b['tpi'][~b['mask']] != 0).any()


def tool():
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import distill as tool_mod
    return tool_mod


def test_the_tool_writes_a_guarded_checkpoint_and_resumes(toy_data, tmp_path, capsys):
    tm = tool()
    tpath = str(tmp_path / 'teacher.pt')
    net.save(teacher(), tpath, {'iter': 7})
    out = str(tmp_path / 'student.pt')
    args = [tpath, out, '--data', str(toy_data / 'data' / 'iter_0001'), '--heldout', str(toy_data / 'data' / 'iter_0002'),
            '--arch', 'tokens', '--hidden', '32', '--blocks', '2', '--heads', '2', '--steps', '200', '--batch', '128',
            '--lr', '2e-3', '--warmup', '20', '--eval-every', '100', '--save-every', '100', '--report-every', '50',
            '--torch-threads', '1', '--normal-priority']
    assert tm.main(args + ['--keep-best']) == 0
    s, extra = net.load(out, game='toy')              # the encoding guard accepts it
    b, bextra = net.load(str(tmp_path / 'student.best.pt'), game='toy')   # --keep-best: the lowest held-out KL
    kls = {h['step']: h['policy_kl'] for h in extra['distill']['heldout']}
    assert bextra['distill']['step'] in (100, 200) and kls[bextra['distill']['step']] == min(kls[100], kls[200])
    assert s.config['arch'] == 'tokens' and s.config['d'] == 32 and s.config['layers'] == 2
    rec = extra['distill']
    assert extra['distilled_from'] == os.path.abspath(tpath) and extra['teacher_extra'] == {'iter': 7}
    assert len(extra['teacher_sha256']) == 64 and rec['settings']['steps'] == 200
    assert [h['step'] for h in rec['heldout']] == [0, 100, 200]
    assert rec['heldout'][-1]['policy_kl'] < rec['heldout'][0]['policy_kl'] / 3
    assert not os.path.exists(out + '.partial')
    log = [json.loads(line) for line in open(out + '.log.jsonl')]
    assert log[0]['event'] == 'start' and log[-1]['event'] == 'done' and log[0]['student_net']['d'] == 32
    # a partial checkpoint resumes at its step
    part = str(tmp_path / 'resumed.pt')
    torch.save({'config': s.config, 'state': s.state_dict(), 'opt': torch.optim.AdamW(s.parameters()).state_dict(),
                'step': 150}, part + '.partial')
    assert tm.main([tpath, part] + args[2:] + ['--resume']) == 0
    lines = [json.loads(line) for line in open(part + '.log.jsonl')]
    assert lines[0]['start'] == 150 and [x['step'] for x in lines if x['event'] == 'train'] == [200]
    # --init: a further distillation starts from the student (its size, its weights)
    more = str(tmp_path / 'more.pt')
    assert tm.main([tpath, more] + args[2:4] + args[4:6] + ['--init', out, '--hidden', '16', '--steps', '1', '--lr', '1e-9',
                                                            '--warmup', '0', '--eval-every', '0', '--normal-priority']) == 0
    m2, e2 = net.load(more, game='toy')
    assert m2.config == s.config and e2['distill']['heldout'][0]['policy_kl'] == pytest.approx(rec['heldout'][-1]['policy_kl'], abs=1e-6)
    # --compare: the student against its teacher on the held-out shards
    capsys.readouterr()
    assert tm.main(['--compare', out, tpath, '--data', str(toy_data / 'data' / 'iter_0002'), '--normal-priority']) == 0
    cmp = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert cmp['policy_kl'] == pytest.approx(rec['heldout'][-1]['policy_kl'], abs=1e-6)


def test_the_tool_refuses_shards_of_another_encoding(toy_data, tmp_path):
    tm = tool()
    tpath = str(tmp_path / 'teacher.pt')
    net.save(teacher(), tpath)
    bad = tmp_path / 'bad' / 'iter_0001'
    sh = data.read_shard(data.shard_paths(str(toy_data / 'data' / 'iter_0002'))[0])
    meta = dict(sh.pop('meta'), features=sh['x'].shape[1] - 1)
    os.makedirs(bad)
    np.savez(str(bad / 'c00_000.npz'), meta=np.array(json.dumps(meta)), **sh)
    with pytest.raises(SystemExit, match='another encoding'):
        tm.main([tpath, str(tmp_path / 's.pt'), '--data', str(toy_data / 'data' / 'iter_0001'), '--data', str(bad),
                 '--steps', '1', '--normal-priority'])


def run_loop(run, *extra):
    smoke = ['--preset', 'toy-smoke', '--conc', '24', '--games', '24', '--eval-games', '8', '--eval-random-games', '4',
             '--elo-games', '4', '--ref-games', '8']
    e = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH)
    return subprocess.run([sys.executable, '-m', 'zero.run', '--run', str(run), *smoke, *extra], cwd=ROOT,
                          capture_output=True, text=True, timeout=300, env=e)


def test_fork_run_net_continues_a_run_with_a_distilled_network(tmp_path):
    """The A/B of a bigger network (docs/az.md 13.3): the run's checkpoint 2 distilled into a bigger
    student on the run's own shards, a continuation of the run at 2 with the student as its
    checkpoint 2, one more iteration trains the student."""
    src = tmp_path / 'src'
    p = run_loop(src, '--iters', '2', '--hidden', '32', '--blocks', '1')
    assert p.returncode == 0, p.stdout + p.stderr
    tm = tool()
    student = str(tmp_path / 'student.pt')
    assert tm.main([str(src / 'models' / 'iter_0002.pt'), student, '--data', str(src / 'data'), '--hidden', '48',
                    '--blocks', '2', '--steps', '60', '--batch', '64', '--warmup', '10', '--eval-every', '0',
                    '--torch-threads', '1', '--normal-priority']) == 0
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import fork_run
    fork = tmp_path / 'fork'
    assert fork_run.main(['--source', str(src), '--dest', str(fork), '--iter', '2', '--net', student]) == 0
    m, extra = net.load(str(fork / 'models' / 'iter_0002.pt'), game='toy')
    s, _ = net.load(student, game='toy')
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), s.state_dict().values()))
    assert extra['iter'] == 2 and extra['net_from'] == os.path.abspath(student) and 'distill' in extra
    assert not (fork / 'models' / 'opt_latest.pt').exists() and (src / 'models' / 'opt_latest.pt').exists()
    cfg = json.load(open(fork / 'config.json'))
    assert (cfg['arch'], cfg['hidden'], cfg['blocks']) == ('mlp', 48, 2)
    res = json.load(open(fork / 'elo' / 'results.json'))
    assert not any('iter_0002' in k.split('|') for k in res['pairs']) and 'iter_0002' not in res.get('logged', [])
    src_res = json.load(open(src / 'elo' / 'results.json'))
    assert any('iter_0002' in k.split('|') for k in src_res['pairs'])          # the source is untouched
    p = run_loop(fork, '--iters', '3')
    assert p.returncode == 0, p.stdout + p.stderr
    m3, _ = net.load(str(fork / 'models' / 'iter_0003.pt'), game='toy')
    assert m3.config['hidden'] == 48 and m3.config['blocks'] == 2
    assert not all(torch.equal(a, b) for a, b in zip(m3.state_dict().values(), m.state_dict().values()))
    bad = str(tmp_path / 'bad.pt')                     # another game's network is refused
    net.save(net.make_net(G.info('toy_team'), 'toy_team', 'mlp', 16, 1), bad)
    with pytest.raises(net.EncodingMismatch):
        fork_run.main(['--source', str(src), '--dest', str(tmp_path / 'fork2'), '--iter', '2', '--net', bad])
