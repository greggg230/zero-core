"""Batched inference for the leaf-batch protocol (docs/search.md 2) and the drive loop.

`NetEvaluator(model, device)` answers a network LeafBatch: the features go through the
network in one call, the policy logits of each leaf's legal actions are gathered and
softmaxed per leaf (a segment softmax on the device), the value head is softmaxed over the
seats present. CPU and CUDA; the same object can take new weights in place (`set_weights`,
rolling self-play). `UniformEvaluator` is the no-network stand-in (tests, iteration 0 checks).

`drive(driver, evaluators, externals)` runs any driver to the end (or to the end of a rolling
phase): network batches go to `evaluators[key]`, external batches to `externals[key]` (one
payload string per request). `check()` is called after every batch (PAUSE: raises).
External batches are answered on worker threads while the other groups keep searching and
evaluating (drivers with `poll_batch`): answering them inline stalled every group for the
external player's latency at each search step. Caverna: 200 games vs the JS bot (~18 ms per
bot decision) 129 s inline -> 110 s async, same games (run.py `ext_groups`, docs/az.md 5); the
Glass Road JS bot (~0.1 s per decision): 12.4 -> 3.75 min.
"""
import time

import numpy as np
import torch

from .protocol import MAX_SEATS


def use_amp(model, device):
    """bfloat16 autocast for a conv network (`net.GridNet`) on CUDA (~1.6-2.6x the leaves/s of
    float32 at batches 512-1024); the MLP keeps float32 (faster as is)."""
    return torch.device(device).type == 'cuda' and getattr(model, 'config', {}).get('arch') == 'grid'


class NetEvaluator:
    def __init__(self, model, device='cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.amp = use_amp(model, self.device)
        self.calls = 0
        self.leaves = 0
        self.seconds = 0.0

    def set_weights(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.inference_mode()
    def evaluate(self, x, n_players, opt_off, opt_action):
        """numpy in -> (priors [options], value [B, MAX_SEATS], margin [B, MAX_SEATS]) numpy."""
        t0 = time.perf_counter()
        dev = self.device
        B = x.shape[0]
        xt = torch.from_numpy(x).to(dev, non_blocking=True)
        nt = torch.from_numpy(n_players.astype(np.int64)).to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp):
            plog, vlog, mar = self.model(xt, nt)
        cnt = torch.from_numpy(np.diff(opt_off).astype(np.int64)).to(dev)
        rows = torch.repeat_interleave(torch.arange(B, device=dev), cnt)
        acts = torch.from_numpy(opt_action.astype(np.int64)).to(dev)
        lg = plog[rows, acts].float()
        mx = torch.full((B,), -torch.inf, device=dev).scatter_reduce(0, rows, lg, 'amax', include_self=True)
        e = torch.exp(lg - mx[rows])
        s = torch.zeros(B, device=dev).index_add_(0, rows, e)
        pri = (e / s[rows]).cpu().numpy().astype(np.float32)
        val = torch.softmax(vlog.float(), dim=1)
        seats = self.model.seats
        value = np.zeros((B, MAX_SEATS), np.float32)
        margin = np.zeros((B, MAX_SEATS), np.float32)
        value[:, :seats] = val.cpu().numpy()
        margin[:, :seats] = mar.float().cpu().numpy()
        value[np.arange(MAX_SEATS)[None, :] >= n_players[:, None]] = 0.0
        self.calls += 1
        self.leaves += B
        self.seconds += time.perf_counter() - t0
        return pri, value, margin

    def __call__(self, lb):
        return self.evaluate(lb.x, lb.n_players, lb.opt_off, lb.opt_action)


class UniformEvaluator:
    """Uniform priors, value 1/n for every seat present, margin 0."""

    def __call__(self, lb):
        off = lb.opt_off
        cnt = np.diff(off)
        pri = np.repeat((1.0 / cnt).astype(np.float32), cnt)
        B = lb.B
        n = lb.n_players.astype(np.float32)[:, None]
        value = ((np.arange(MAX_SEATS)[None, :] < n) / n).astype(np.float32)
        return pri, value, np.zeros((B, MAX_SEATS), np.float32)


def drive(driver, evaluators, externals=None, check=None, on_batch=None, ext_threads=16):
    """Run `driver` until next_batch() returns None. -> number of batches."""
    externals = externals or {}
    if externals and hasattr(driver, 'poll_batch'):
        return _drive_async(driver, evaluators, externals, check, on_batch, ext_threads)
    nb = 0
    while True:
        lb = driver.next_batch()
        if lb is None:
            return nb
        if lb.external:
            driver.submit_external(lb.id, _external(externals, lb)(lb))
        else:
            driver.submit(lb.id, *evaluators[lb.key](lb))
        nb += 1
        if on_batch is not None:
            on_batch(lb)
        if check is not None:
            check()


def _external(externals, lb):
    ans = externals.get(lb.key)
    if ans is None:
        raise KeyError(f'no external player for key {lb.key!r} (have {sorted(externals)})')
    return ans


def _drive_async(driver, evaluators, externals, check, on_batch, ext_threads):
    """`drive` with external batches answered on worker threads: while a group waits for its
    external answers the driver searches and evaluates the other groups (`poll_batch`), and
    answers are submitted as they arrive. Same batches, same answers (each external player's
    seeds depend on the game and its request count, not on timing)."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    nb = 0
    pending = {}                                   # batch id -> future of its payloads
    ex = ThreadPoolExecutor(max_workers=ext_threads, thread_name_prefix='zero-ext')
    try:
        while True:
            for bid in [b for b, f in pending.items() if f.done()]:
                driver.submit_external(bid, pending.pop(bid).result())
            lb = driver.poll_batch() if pending else driver.next_batch()
            if lb is None:
                if not pending:
                    return nb
                wait(list(pending.values()), timeout=0.002, return_when=FIRST_COMPLETED)
                continue
            if lb.external:
                pending[lb.id] = ex.submit(_external(externals, lb), lb)
            else:
                driver.submit(lb.id, *evaluators[lb.key](lb))
            nb += 1
            if on_batch is not None:
                on_batch(lb)
            if check is not None:
                check()
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
