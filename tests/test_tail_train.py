"""Training a network with gates (docs/az.md 43.6): the run logs the gate values (`train.tail.gates`, by segment) and the
steps whose batches held each gated segment's tokens, warns when a gate is still exactly 0 after GATE_STUCK_STEPS of
them, and keeps the gates out of AdamW's weight decay (their own optimizer group, last; every other group as before)."""
import copy
import os

import pytest
import torch

from zero import package
from zero import run as RUN
from zero.net import TokenNet, save, widen

from test_tail_loop import EchoEval, cfg
from test_tail_net import TOY_GATED, TOY_TAIL_TOKENS_BASE


def gated_run(tmp, theta, **kw):
    """A tiny toy_tail run (lane X's `test_tail_loop.cfg`) whose checkpoint 0 is a token network with the marks gated
    (a widen of one without them: gate 0, then set to `theta`), its iteration 1 self-played by the row echo."""
    torch.manual_seed(3)
    m = TokenNet(72, 16, TOY_TAIL_TOKENS_BASE, d=16, layers=1, heads=2, ff=32, game='toy_tail')
    w = widen(m, layout=TOY_GATED)
    with torch.no_grad():
        w.gate.data[:] = theta
    R = RUN.Run(str(tmp), cfg(hidden=16, blocks=1, heads=2, **kw))
    os.makedirs(os.path.join(str(tmp), 'models'), exist_ok=True)
    save(w, RUN.model_path(str(tmp), 0), {'iter': 0})
    R.evaluator = lambda path: EchoEval()
    R.check_rows()
    out = R._selfplay(1)
    return R, out


def test_training_logs_the_gates_and_never_decays_them(tmp_path, monkeypatch):
    monkeypatch.setitem(package.CORE.tokens, 'toy_tail', TOY_GATED)
    R, out = gated_run(tmp_path, 0.3, wd=0.5)
    model, ost, stats = R.train_model(1, out['samples'])
    s = stats()
    t = s['tail']
    assert set(t['gates']) == {'marks'} and t['gates']['marks'] > 0 and t['gate_steps']['marks'] > 0
    assert 'gates_stuck' not in t
    groups = ost['param_groups']
    assert groups[-1]['weight_decay'] == 0.0 and len(groups[-1]['params']) == 1
    assert all(g['weight_decay'] == 0.5 for g in groups[:-1])
    # (the gate's own moments: it was stepped in its group)
    assert ost['state'][groups[-1]['params'][0]]['exp_avg'].shape == (1,)


def test_a_gate_still_shut_is_said_loudly(tmp_path, monkeypatch):
    """A present gate still exactly 0 after GATE_STUCK_STEPS steps that held its tokens: `gates_stuck` and a warning in
    the run's log (here lr 0: the gate cannot move)."""
    monkeypatch.setitem(package.CORE.tokens, 'toy_tail', TOY_GATED)
    monkeypatch.setattr(RUN, 'GATE_STUCK_STEPS', 3)
    R, out = gated_run(tmp_path, 0.0, lr=0.0, lr_min=0.0)
    model, ost, stats = R.train_model(1, out['samples'])
    t = stats()['tail']
    assert t['gates'] == {'marks': 0.0} and t['gate_steps']['marks'] >= 3 and t['gates_stuck'] == ['marks']
    log = open(os.path.join(str(tmp_path), 'log.txt'), encoding='utf-8').read()
    assert 'WARNING gate marks still exactly 0' in log, log[-2000:]


def test_a_network_without_gates_keeps_the_optimizer_of_before(tmp_path):
    """No gate: no group of its own (the groups, their decay and their order as before)."""
    R = RUN.Run(str(tmp_path), cfg(hidden=16, blocks=1, heads=2))
    torch.manual_seed(4)
    m = TokenNet(72, 16, package.TOY_TAIL_TOKENS, d=16, layers=1, heads=2, ff=32, game='toy_tail')
    os.makedirs(os.path.join(str(tmp_path), 'models'), exist_ok=True)
    save(m, RUN.model_path(str(tmp_path), 0), {'iter': 0})
    R.evaluator = lambda path: EchoEval()
    out = R._selfplay(1)
    _, ost, _ = R.train_model(1, out['samples'])
    assert len(ost['param_groups']) == 1 and ost['param_groups'][0]['weight_decay'] == R.cfg['wd']


def _loaded_states(monkeypatch):
    """Every optimizer state `torch.optim.AdamW.load_state_dict` loads, as the optimizer holds it right after."""
    got = []
    orig = torch.optim.AdamW.load_state_dict

    def load(self, state):
        orig(self, state)
        got.append(copy.deepcopy(self.state_dict()))       # (the training steps after it update it in place)
    monkeypatch.setattr(torch.optim.AdamW, 'load_state_dict', load)
    return got


def _same_state(a, b):
    """Two optimizer state dicts with the same groups (hyper-parameters, parameter counts) and every moment equal."""
    ga, gb = a['param_groups'], b['param_groups']
    assert [len(g['params']) for g in ga] == [len(g['params']) for g in gb]
    assert [g['weight_decay'] for g in ga] == [g['weight_decay'] for g in gb]
    ia = [i for g in ga for i in g['params']]
    ib = [i for g in gb for i in g['params']]
    for i, j in zip(ia, ib):
        sa, sb = a['state'][i], b['state'][j]
        assert set(sa) == set(sb)
        for k in sa:
            assert torch.equal(torch.as_tensor(sa[k]).cpu(), torch.as_tensor(sb[k]).cpu()), (i, k)


def test_a_gated_networks_optimizer_state_reloads_exactly(tmp_path, monkeypatch):
    """A network with gates (their group last, without decay) trains iteration 2 from iteration 1's optimizer state, its
    groups and moments exactly as saved - in the same process and through a resume (a new Run for the directory, as a
    restarted run) - never a fresh optimizer (docs/az.md 43.10)."""
    monkeypatch.setitem(package.CORE.tokens, 'toy_tail', TOY_GATED)
    R, out = gated_run(tmp_path, 0.3, wd=0.5)
    R.train(1, out['samples'])
    saved = torch.load(str(tmp_path / 'models' / 'opt_latest.pt'), weights_only=False)
    assert saved['iter'] == 1 and saved['state']['param_groups'][-1]['weight_decay'] == 0.0
    got = _loaded_states(monkeypatch)
    out2 = R._selfplay(2)
    R.train_model(2, out2['samples'])
    assert len(got) == 1
    _same_state(saved['state'], got[0])
    # through a resume: a new process's Run reads the state from disk
    R2 = RUN.Run(str(tmp_path), R.cfg)
    R2.train_model(2, out2['samples'])
    assert len(got) == 2
    _same_state(saved['state'], got[1])
    log = open(os.path.join(str(tmp_path), 'log.txt'), encoding='utf-8').read() if os.path.exists(
        os.path.join(str(tmp_path), 'log.txt')) else ''
    assert 'fresh optimizer' not in log


def test_an_optimizer_state_that_does_not_fit_is_refused(tmp_path, monkeypatch):
    """The state saved with the last checkpoint not fitting its network is refused (OptimizerStateMismatch: an
    EncodingMismatch, exit 78), never a fresh optimizer in silence; deleting it starts one deliberately; a state of
    another iteration is said and replaced by a fresh one."""
    monkeypatch.setitem(package.CORE.tokens, 'toy_tail', TOY_GATED)
    R, out = gated_run(tmp_path, 0.3)
    R.train(1, out['samples'])
    path = str(tmp_path / 'models' / 'opt_latest.pt')
    saved = torch.load(path, weights_only=False)
    bad = {'iter': saved['iter'], 'state': {'param_groups': saved['state']['param_groups'],
                                             'state': {i: {k: (v[..., :1] if torch.is_tensor(v) and v.dim() else v)
                                                           for k, v in st.items()}
                                                       for i, st in saved['state']['state'].items()}}}
    torch.save(bad, path)
    out2 = R._selfplay(2)
    with pytest.raises(RUN.OptimizerStateMismatch, match='does not fit'):
        R.train_model(2, out2['samples'])
    assert issubclass(RUN.OptimizerStateMismatch, RUN.EncodingMismatch)
    os.remove(path)
    R.train_model(2, out2['samples'])                # (deliberately fresh)
    torch.save(dict(saved, iter=0), path)
    R.train_model(2, out2['samples'])
    log = open(os.path.join(str(tmp_path), 'log.txt'), encoding='utf-8').read()
    assert 'the optimizer state is of iteration 0, not 1: fresh optimizer' in log
