"""Training a network with gates (docs/az.md 43.6): the run logs the gate values (`train.tail.gates`, by segment) and the
steps whose batches held each gated segment's tokens, warns when a gate is still exactly 0 after GATE_STUCK_STEPS of
them, and keeps the gates out of AdamW's weight decay (their own optimizer group, last; every other group as before)."""
import os

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
