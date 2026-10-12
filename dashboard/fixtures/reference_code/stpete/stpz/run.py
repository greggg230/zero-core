"""AlphaZero loop: self-play -> train -> evaluate, repeated.

  python -m stpz.run --run runs/r1 [--iters 60] [--games 4000] [--workers 16] ...

Iteration i self-plays with models/iter_{i-1}.pt, trains on the last --window
iterations of data, writes models/iter_i.pt and appends a line to metrics.jsonl.
Resumes from the newest checkpoint. Create a file named STOP in the run dir to end
cleanly after the current iteration.
"""
import argparse
import glob
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .net import Net, save, load
from . import workers
from . import elo
from .mcts import check_policy_target
from .targets import summary as target_summary

DEFAULTS = dict(
    iters=60, games=4000, players=[2], sims=128, sims_fast=32, p_full=0.25, conc=128,
    c_puct=1.0, alpha=0.4, eps=0.25, temp_rounds=2,
    workers=16, window=8, min_window_samples=200_000, batch=1024, reuse=4.0, lr=1e-3,
    wd=1e-4, score_w=0.5, hidden=256, blocks=4, infer_device='cpu', lr_halve_every=30, lr_min=1e-4,
    eval_games=400, eval_sims=64, eval_every=1, keep_data=10, elo_games=200, phase_timeout=3600,
    # value_q: weight of the search's own root win estimate in the value target (0 = game result only)
    # margin_w: search utility = P(win) + margin_w * tanh(score margin / 20)
    # chance_k: sampled draws kept per chance node (0 = resample on every visit)
    value_q=0.0, margin_w=0.0, chance_k=0,
    driver='python',            # 'rust': self-play and network-vs-network games on stpz_rs
    # infer_device 'auto' uses the GPU only with >= gpu_min_free_mb free and <= gpu_max_util % busy
    gpu_min_free_mb=12000, gpu_max_util=40,
    # optional fixed reference opponent (e.g. another run's checkpoint) with its own search settings
    ref_model='', ref_games=400, ref_chance_k=0, ref_margin_w=0.0,
    # policy target of self-play records: 'visits' (root visit distribution) or 'cq' (Gumbel
    # AlphaZero's completed-Q target over the root decider's P(win): softmax(ln P0 +
    # (cq_c_visit + max N) * cq_c_scale * completedQ)); evaluation games record nothing
    policy_target='visits', cq_c_visit=50.0, cq_c_scale=1.0,
)


class Pool:
    """Worker processes for self-play and evaluation games.

    * A worker that dies (BrokenProcessPool) costs only its unfinished tasks: the pool is
      rebuilt and those tasks are retried, up to `retries` times per phase, before the run
      gives up (and tools/supervise.py restarts it from the last checkpoint).
    * A phase running past `timeout` seconds raises TimeoutError (same supervisor path).
    * When the inference device changes from GPU to CPU the workers are replaced, so their
      GPU memory is handed back."""

    def __init__(self, n, timeout, logf=print, retries=3):
        self.n, self.timeout, self.logf, self.retries = n, timeout, logf, retries
        self.device = None
        self._new()

    def _new(self):
        self.ex = ProcessPoolExecutor(self.n, mp_context=mp.get_context('spawn'))

    def use_device(self, device):
        if self.device == 'cuda' and device != 'cuda':
            self.restart()
        self.device = device

    def map(self, fn, tasks, chunksize=1):
        results = [None] * len(tasks)
        pending = list(range(len(tasks)))
        deadline = time.time() + self.timeout
        for attempt in range(self.retries + 1):
            futs = {self.ex.submit(fn, tasks[i]): i for i in pending}
            try:
                for f in as_completed(futs, timeout=max(1.0, deadline - time.time())):
                    results[futs[f]] = f.result()
                return results
            except BrokenProcessPool:
                for f, i in futs.items():
                    if results[i] is None and f.done() and not f.cancelled() and f.exception() is None:
                        results[i] = f.result()
                pending = [i for i in pending if results[i] is None]
                if attempt == self.retries:
                    raise
                self.logf(f'a worker died; retrying {len(pending)} of {len(tasks)} tasks with a fresh pool '
                          f'(attempt {attempt + 2})')
                self.restart()

    def restart(self):
        self.kill()
        self._new()

    def kill(self):
        import psutil
        for k in psutil.Process().children(recursive=True):
            try:
                k.kill()
            except psutil.NoSuchProcess:
                pass
        self.ex.shutdown(wait=False, cancel_futures=True)


def phase_device(cfg):
    """infer_device 'auto': the GPU when it has room and is not busy with something else
    (checked between phases, when this run's workers are not using it), else the CPU."""
    dev = cfg['infer_device']
    if dev != 'auto':
        return dev
    try:
        import subprocess
        used = util = 0
        for _ in range(3):
            out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                                  '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10).stdout
            u, total, ut = (float(x) for x in out.strip().splitlines()[0].split(','))
            used, util = max(used, u), max(util, ut)
            time.sleep(0.3)
        if total - used >= cfg['gpu_min_free_mb'] and util <= cfg['gpu_max_util']:
            return 'cuda'
    except Exception:
        pass
    return 'cpu'


def log(run, msg):
    line = time.strftime('%H:%M:%S ') + msg
    print(line, flush=True)
    with open(os.path.join(run, 'log.txt'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def model_path(run, i):
    return os.path.join(run, 'models', f'iter_{i:04d}.pt')


def latest_iter(run):
    its = [int(os.path.basename(p)[5:9]) for p in glob.glob(os.path.join(run, 'models', 'iter_*.pt'))]
    return max(its) if its else None


def selfplay(pool, run, cfg, it):
    d = os.path.join(run, 'data', f'iter_{it:04d}')
    os.makedirs(d, exist_ok=True)
    n = cfg['workers'] * 2                         # 2 tasks per worker for load balance
    per = [cfg['games'] // n + (1 if k < cfg['games'] % n else 0) for k in range(n)]
    tasks = [dict(model=model_path(run, it - 1), out=os.path.join(d, f'c{k:02d}.npz'),
                  games=per[k], seed=it * 1000 + k,
                  device=pool.device, driver=cfg['driver'],
                  **{k2: cfg[k2] for k2 in ('players', 'sims', 'sims_fast', 'p_full', 'conc',
                                            'c_puct', 'alpha', 'eps', 'temp_rounds',
                                            'chance_k', 'margin_w', 'policy_target', 'cq_c_visit',
                                            'cq_c_scale')})
             for k in range(n) if per[k]]
    t0 = time.time()
    stats = pool.map(workers.selfplay_task, tasks, chunksize=1)
    tot = {k: sum(s[k] for s in stats) for k in stats[0]}
    fin = tot['games'] - tot['unfinished']
    return {'games': tot['games'], 'unfinished': tot['unfinished'], 'samples': tot['samples'],
            'decisions_per_game': tot['decisions'] / max(1, tot['games']),
            'rounds': tot['rounds'] / max(1, fin), 'avg_score': tot['score'] / max(1, fin),
            'seconds': time.time() - t0, 'target': target_summary(tot, cfg['policy_target'])}


def load_window(run, it, cfg, device, nfeat):
    """Most recent data dirs, at least `window` iterations and min_window_samples. Feature
    rows are cut/zero-padded to the network's input width; data without the search's win
    estimate ('q') uses the game result in its place."""
    dirs = sorted(glob.glob(os.path.join(run, 'data', 'iter_*')), reverse=True)
    parts, n = [], 0
    for k, d in enumerate(dirs):
        if k >= cfg['window'] and n >= cfg['min_window_samples']:
            break
        for f in sorted(glob.glob(os.path.join(d, '*.npz'))):
            z = np.load(f)
            part = {key: z[key] for key in ('x', 'pi', 'mask', 'z', 'sc')}
            part['q'] = z['q'] if 'q' in z.files else part['z']
            x = part['x']
            if x.shape[1] > nfeat:
                part['x'] = x[:, :nfeat]
            elif x.shape[1] < nfeat:
                part['x'] = np.pad(x, ((0, 0), (0, nfeat - x.shape[1])))
            parts.append(part)
            n += len(part['x'])
    cat = lambda key: torch.from_numpy(np.concatenate([p[key] for p in parts])).to(device)
    return {key: cat(key) for key in ('x', 'pi', 'mask', 'z', 'sc', 'q')}, n


def lr_at(cfg, it):
    """Step decay: halve every lr_halve_every iterations, never below lr_min."""
    return max(cfg['lr_min'], cfg['lr'] * 0.5 ** (it // cfg['lr_halve_every']))


def train(run, cfg, it, new_samples, device):
    """Train on the GPU; if it is out of memory (it is shared with ComfyUI and games),
    fall back to the CPU for this iteration instead of crashing."""
    if device == 'cuda':
        try:
            return _train(run, cfg, it, new_samples, 'cuda')
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(run, f'iter {it}: GPU out of memory, training on CPU this iteration')
    torch.set_num_threads(8)
    return _train(run, cfg, it, new_samples, 'cpu')


def _train(run, cfg, it, new_samples, device):
    prev = model_path(run, it - 1)
    model, _ = load(prev, device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    opt_path = os.path.join(run, 'models', 'opt_latest.pt')
    if os.path.exists(opt_path):
        try:
            opt.load_state_dict(torch.load(opt_path, map_location=device))
        except Exception:
            pass
    lr = lr_at(cfg, it)
    for g in opt.param_groups:
        g['lr'] = lr
    data, n = load_window(run, it, cfg, device, model.features)
    vq = cfg['value_q']
    steps = max(50, math.ceil(new_samples * cfg['reuse'] / cfg['batch']))
    t0 = time.time()
    acc = {'policy': 0.0, 'value': 0.0, 'score': 0.0, 'entropy': 0.0}
    for step in range(steps):
        idx = torch.randint(n, (cfg['batch'],), device=device)
        x = data['x'][idx].float()
        pi = data['pi'][idx].float()
        mask = data['mask'][idx]
        z = data['z'][idx]
        if vq:
            z = (1 - vq) * z + vq * data['q'][idx]
        sc = data['sc'][idx]
        plog, vlog, spred = model(x)
        plog = plog.masked_fill(~mask, -1e9)
        logp = F.log_softmax(plog, dim=1)
        lp = -(pi * logp).sum(1).mean()
        lv = -(z * F.log_softmax(vlog, dim=1).clamp(min=-1e4)).sum(1).mean()
        present = (x[:, ::model_seat_stride()][:, :4] > 0.5).float()
        ls = (((spred - sc) ** 2) * present).sum(1).mean() / 2
        loss = lp + lv + cfg['score_w'] * ls
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step >= steps - min(steps, 200):
            with torch.no_grad():
                p = logp.exp()
                ent = -(p * logp.clamp(min=-30)).sum(1).mean()
            k = min(steps, 200)
            acc['policy'] += lp.item() / k
            acc['value'] += lv.item() / k
            acc['score'] += ls.item() / k
            acc['entropy'] += ent.item() / k
    model.eval()
    save(model, model_path(run, it), {'iter': it})
    torch.save(opt.state_dict(), opt_path + '.tmp')
    os.replace(opt_path + '.tmp', opt_path)
    del data
    if device == 'cuda':
        torch.cuda.empty_cache()
    return dict(acc, steps=steps, window_samples=n, seconds=time.time() - t0, lr=lr, device=device)


def model_seat_stride():
    from .encode import SEAT_F
    return SEAT_F


def evaluate(pool, run, cfg, it, opp, games, sims, opp_model=None, seed=0, players=2, opp_search=None):
    """opp_search: (chance_k, margin_w) for a network opponent from another run."""
    n = min(cfg['workers'] * 2, max(1, games // 16))
    per = [games // n + (1 if k < games % n else 0) for k in range(n)]
    tasks = []
    off = 0
    for k in range(n):
        tasks.append(dict(model=model_path(run, it), opp=opp, games=per[k],
                          seed=seed + 7919 * it + k, sims=sims, conc=min(cfg['conc'], per[k]),
                          c_puct=cfg['c_puct'], first_seat=off % players, opp_model=opp_model,
                          device=pool.device, players=players, driver=cfg['driver'],
                          chance_k=cfg['chance_k'], margin_w=cfg['margin_w']))
        if opp_search:
            tasks[-1].update(opp_chance_k=opp_search[0], opp_margin_w=opp_search[1])
        off += per[k]
    rows = [r for rs in pool.map(workers.eval_task, tasks, chunksize=1) for r in rs]
    w = sum(r[0] for r in rows) / len(rows)
    se = math.sqrt(max(w * (1 - w), 1e-9) / len(rows))
    return {'win': round(w, 4), 'se': round(se, 4), 'margin': round(sum(r[1] for r in rows) / len(rows), 2),
            'games': len(rows), 'score': sum(r[0] for r in rows)}


def read_metrics(run):
    p = os.path.join(run, 'metrics.jsonl')
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def elo_update(pool, run, cfg, latest, current=None):
    """Play any missing rating matches up to checkpoint `latest`, refit, tag checkpoints.
    `current` is this iteration's metrics dict (not yet in metrics.jsonl)."""
    res = elo.load_results(run)
    elo.baseline_games(res)
    logged = res.setdefault('shrewd_logged', [])
    for m in read_metrics(run) + ([current] if current else []):
        name = elo.ckpt_name(m['iter'])
        if 'eval_shrewd' in m and name not in logged:
            e = m['eval_shrewd']
            elo.add(res, name, elo.ANCHOR, e.get('score', e['win'] * e['games']), e['games'])
            logged.append(name)
    elo.save_results(run, res)
    todo = [(k, k - off) for k in range(1, latest + 1) for off in elo.OFFSETS
            if k - off >= 0 and os.path.exists(model_path(run, k - off))
            and not elo.has(res, elo.ckpt_name(k), elo.ckpt_name(k - off))]
    if todo:
        # all missing matches in one dispatch, split so the tasks fill the workers in one round
        tasks, owners = [], []
        parts = max(1, cfg['workers'] // len(todo))
        for k, j in todo:
            n = cfg['elo_games']
            off = 0
            for q in range(parts):
                sz = n // parts + (1 if q < n % parts else 0)
                tasks.append(dict(model=model_path(run, k), opp='nn', opp_model=model_path(run, j), games=sz,
                                  seed=31 + 7919 * k + 101 * j + q, sims=cfg['eval_sims'], conc=sz,
                                  c_puct=cfg['c_puct'], first_seat=off % 2, device=pool.device, driver=cfg['driver'],
                                  chance_k=cfg['chance_k'], margin_w=cfg['margin_w']))
                owners.append((k, j))
                off += sz
        agg = {}
        for kj, rows in zip(owners, pool.map(workers.eval_task, tasks, chunksize=1)):
            sc, n = agg.get(kj, (0.0, 0))
            agg[kj] = (sc + sum(r[0] for r in rows), n + len(rows))
        for (k, j), (sc, n) in agg.items():
            elo.add(res, elo.ckpt_name(k), elo.ckpt_name(j), sc, n)
            log(run, f'elo match {elo.ckpt_name(k)} vs {elo.ckpt_name(j)}: {sc:.1f}/{n}')
        elo.save_results(run, res)
    ratings = elo.fit(res)
    elo.write_ratings(run, ratings)
    best = elo.tag_checkpoints(run, ratings, latest)
    return ratings, best


def prune(run, it, cfg):
    for d in glob.glob(os.path.join(run, 'data', 'iter_*')):
        k = int(os.path.basename(d)[5:9])
        if k <= it - cfg['keep_data']:
            shutil.rmtree(d, ignore_errors=True)


def check_config(cfg):
    """Reject settings the run cannot honour before anything starts (SystemExit). With the
    Rust driver, returns the stpz_rs build in use (logged), which must support the policy
    target: an older build would refuse a 'cq' seat in every worker."""
    try:
        check_policy_target(cfg['policy_target'], cfg['cq_c_visit'], cfg['cq_c_scale'])
    except ValueError as e:
        raise SystemExit(f'bad settings: {e}')
    if cfg['driver'] != 'rust':
        return None
    import stpz_rs
    have = getattr(stpz_rs, 'POLICY_TARGETS', ('visits',))
    if cfg['policy_target'] not in have:
        raise SystemExit(f'the installed stpz_rs ({stpz_rs.__file__}) has no policy_target '
                         f'{cfg["policy_target"]!r}: build and install rust/ from this checkout')
    return f'{stpz_rs.__file__} (policy targets {list(have)})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    for k, v in DEFAULTS.items():
        if isinstance(v, list):
            ap.add_argument('--' + k.replace('_', '-'), type=int, nargs='+', default=None)
        else:
            ap.add_argument('--' + k.replace('_', '-'), type=type(v), default=None)
    args = ap.parse_args()
    run = args.run
    os.makedirs(os.path.join(run, 'models'), exist_ok=True)
    cfg_path = os.path.join(run, 'config.json')
    cfg = dict(DEFAULTS)
    if os.path.exists(cfg_path):
        cfg.update(json.load(open(cfg_path)))
    for k in DEFAULTS:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    rs_build = check_config(cfg)
    json.dump(cfg, open(cfg_path, 'w'), indent=1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    workers.below_normal()

    start = latest_iter(run)
    if start is None:
        torch.manual_seed(0)
        save(Net(cfg['hidden'], cfg['blocks']).eval(), model_path(run, 0), {'iter': 0})
        start = 0
    log(run, f'run {run} starting after iter {start}; cfg {json.dumps(cfg)}')
    if rs_build:
        log(run, f'stpz_rs: {rs_build}')
    stop = os.path.join(run, 'STOP')
    pool = Pool(cfg['workers'], cfg['phase_timeout'], logf=lambda msg: log(run, msg))
    pool.use_device(phase_device(cfg))
    try:
        if start == 0 and not os.path.exists(os.path.join(run, 'metrics.jsonl')):
            base = evaluate(pool, run, cfg, 0, 'shrewd', cfg['eval_games'], cfg['eval_sims'])
            m0 = {'iter': 0, 'eval_shrewd': base, 'time': time.time()}
            if cfg['ref_model']:
                m0['eval_ref'] = evaluate(pool, run, cfg, 0, 'nn', cfg['ref_games'], cfg['eval_sims'],
                                          opp_model=cfg['ref_model'], seed=300,
                                          opp_search=(cfg['ref_chance_k'], cfg['ref_margin_w']))
                m0['eval_ref']['ref'] = cfg['ref_model']
                log(run, f'iter 0 vs reference {cfg["ref_model"]}: {m0["eval_ref"]}')
            open(os.path.join(run, 'metrics.jsonl'), 'a').write(json.dumps(m0) + '\n')
            log(run, f'iter 0 (starting network) vs shrewd: {base}')
        ratings, best = elo_update(pool, run, cfg, start)
        if best:
            log(run, f'elo: best {best["file"]} {best["elo"]:.0f} ±{1.96 * best["se"]:.0f}')
        for it in range(start + 1, cfg['iters'] + 1):
            if os.path.exists(stop):
                log(run, 'STOP file found; exiting')
                break
            t0 = time.time()
            dev = phase_device(cfg)
            if dev != pool.device:
                log(run, f'iter {it}: inference device {pool.device} -> {dev}')
            pool.use_device(dev)
            sp = selfplay(pool, run, cfg, it)
            sp['device'] = dev
            log(run, f'iter {it} selfplay: {sp}')
            tr = train(run, cfg, it, sp['samples'], device)
            log(run, f'iter {it} train: {tr}')
            m = {'iter': it, 'selfplay': sp, 'train': tr}
            if it % cfg['eval_every'] == 0:
                m['eval_shrewd'] = evaluate(pool, run, cfg, it, 'shrewd', cfg['eval_games'], cfg['eval_sims'])
                log(run, f'iter {it} vs shrewd (sims {cfg["eval_sims"]}): {m["eval_shrewd"]}')
                if cfg['ref_model']:
                    m['eval_ref'] = evaluate(pool, run, cfg, it, 'nn', cfg['ref_games'], cfg['eval_sims'],
                                             opp_model=cfg['ref_model'], seed=300,
                                             opp_search=(cfg['ref_chance_k'], cfg['ref_margin_w']))
                    m['eval_ref']['ref'] = cfg['ref_model']
                    log(run, f'iter {it} vs reference {cfg["ref_model"]}: {m["eval_ref"]}')
                for n in sorted(set(cfg['players']) - {2}):
                    m[f'eval_shrewd_{n}p'] = evaluate(pool, run, cfg, it, 'shrewd', cfg['eval_games'] // 2,
                                                      cfg['eval_sims'], players=n, seed=500 + n)
                    log(run, f'iter {it} {n}-player vs {n - 1} shrewd (fair share {1 / n:.2f}): {m[f"eval_shrewd_{n}p"]}')
            ratings, best = elo_update(pool, run, cfg, it, m)
            me = ratings.get(elo.ckpt_name(it))
            if me:
                m['elo'] = {'elo': round(me[0], 1), 'se': round(me[1], 1)}
                log(run, f'iter {it} elo {me[0]:.0f} ±{1.96 * me[1]:.0f}; best {best["file"]} {best["elo"]:.0f}')
            m['seconds'] = time.time() - t0
            m['time'] = time.time()
            with open(os.path.join(run, 'metrics.jsonl'), 'a') as f:
                f.write(json.dumps(m) + '\n')
            prune(run, it, cfg)
    except BaseException as e:
        log(run, f'run aborted: {type(e).__name__}: {e}')
        pool.kill()
        raise
    pool.ex.shutdown()
    log(run, 'done')


if __name__ == '__main__':
    main()
