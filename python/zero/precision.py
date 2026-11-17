"""Network arithmetic precision on CUDA: the run settings `infer_precision` and `train_tf32` (docs/az.md 57).

* `infer_precision` 'fp32' (default: the evaluators of before) | 'tf32' | 'bf16': the run's own evaluators
  (self-play, its league seats, its evaluation and Elo matches; zero.h2h's `precision`) on CUDA. 'bf16' = bfloat16
  autocast around the network's forward (a token network's CUDA graphs captured under it: `TokenGraphRunner(amp=True)`);
  'tf32' = float32 matmuls as TF32 (`matmul_tf32`) while a graph is captured (a replay keeps the kernels it was
  captured with) and around every eager forward. A CPU / MPS evaluator stays float32 (said on stderr).
* `train_tf32` 0 (default) | 1: TF32 matmuls around `Run.train_model`'s step (forward, backward, optimizer step).

The TF32 switch is torch's process-wide `torch.backends.cuda.matmul.allow_tf32` (cuBLAS reads it when a matmul is
issued, from any thread), not a thread-local like autocast: a run trains while its local linger self-plays on another
thread (docs/az.md 5.3). So every block that sets it holds one process lock (`matmul_tf32`), and a run with either
setting runs its training step inside `matmul_tf32(train_tf32)` - its own value, whatever an evaluator's capture on
another thread would set. A run with neither setting never takes the lock (the code of before).
"""
import contextlib
import threading

PRECISIONS = ('fp32', 'tf32', 'bf16')

_LOCK = threading.RLock()


def check(p, what='infer_precision'):
    """`p` as one of PRECISIONS (None / '' -> 'fp32'), else ValueError."""
    if p is None or p == '':
        return 'fp32'
    s = str(p).strip().lower()
    if s not in PRECISIONS:
        raise ValueError(f"{what} {p!r}: one of {', '.join(PRECISIONS)} (docs/az.md 57)")
    return s


@contextlib.contextmanager
def matmul_tf32(on):
    """CUDA float32 matmuls as TF32 (`on`) or IEEE float32 (not `on`) for the block, the flag restored after; the block
    holds the process lock every such block takes (module docstring), so no other thread's block sees this one's
    flag."""
    import torch
    m = torch.backends.cuda.matmul
    with _LOCK:
        old = m.allow_tf32
        m.allow_tf32 = bool(on)
        try:
            yield
        finally:
            m.allow_tf32 = old


def train_scope(cfg, device):
    """The training step's context of run settings `cfg` on `device`: `matmul_tf32(train_tf32)` when the run sets
    train_tf32 (CUDA) or infer_precision 'tf32' (its evaluators may set the flag on another thread), else a null
    context (the step of before)."""
    import torch
    tf32 = bool(cfg.get('train_tf32')) and torch.device(device).type == 'cuda'
    if tf32 or check(cfg.get('infer_precision')) == 'tf32':
        return matmul_tf32(tf32)
    return contextlib.nullcontext()
