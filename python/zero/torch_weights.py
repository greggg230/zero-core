"""`torch-weights/1`: a generation's network as the farm's torch worker loads it (docs/az.md 80; zero-farm's per-machine
torch runtime). A model variant (zero-farm PROTOCOL rev 20, 14) of form 'weights' (the run setting `pool_variants`
'torch=weights', zero.farm_pool): the checkpoint's network - its config and weights - without the optimizer state or
anything else of the checkpoint, plus the run's own inference settings, so a worker plays the gen exactly as the
trainer's local self-play does (`NetEvaluator` with the run's `infer_precision`, `infer_token_chunks`,
`infer_workers`; `torch_threads`) - those settings are trainer-class (check-config never puts them in a self-play
template), so they travel with the network.

The file is a `torch.save` of a dict of plain values and tensors only, loadable with `torch.load(weights_only=True)`
(no code runs when a worker reads it):

    {"format": "torch-weights/1", "net_config": the checkpoint's network config, "state_dict": {name: tensor},
     "aux_older": [[...], ...] (only when the checkpoint's auxiliary head was widened: its older layouts),
     "infer": {"precision": 'fp32'|'bf16'|'tf32', "token_chunks": int, "workers": int, "torch_threads": int},
     "gen": the generation (int or None)}

`load` rebuilds the network as `zero.net.load` does a checkpoint (the encoding guard first, the auxiliary head
widened to the game's targets), so the weights a worker plays are the trainer's bit for bit.
"""
import os

import torch

FORMAT = 'torch-weights/1'
INFER_KEYS = ('precision', 'token_chunks', 'workers', 'torch_threads')


def infer_of(cfg):
    """The run's inference settings a torch worker plays with (module docs)."""
    from .precision import check
    return {'precision': check(cfg.get('infer_precision', 'fp32')), 'token_chunks': int(cfg.get('infer_token_chunks', 0) or 0),
            'workers': int(cfg.get('infer_workers', 1) or 1), 'torch_threads': int(cfg.get('torch_threads', 1) or 1)}


def export(ckpt, dst, cfg=None, gen=None):
    """Checkpoint `ckpt` (zero.net.save's file) -> `dst` as torch-weights/1 (written atomically); `cfg`: the run's
    settings (their `infer_*` / `torch_threads`; None: fp32 defaults). -> dst."""
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    extra = ck.get('extra') or {}
    doc = {'format': FORMAT, 'net_config': ck['config'], 'state_dict': {k: v.detach().cpu().contiguous() for k, v in ck['state'].items()},
           'infer': infer_of(cfg or {}), 'gen': None if gen is None else int(gen)}
    if extra.get('aux_older'):
        doc['aux_older'] = extra['aux_older']
    tmp = f'{dst}.{os.getpid()}.tmp'
    torch.save(doc, tmp)
    os.replace(tmp, dst)
    # (what a worker will do with it: weights_only, so a config the safe unpickler refuses fails here, not on a farm PC)
    read(dst)
    return dst


class NotTorchWeights(ValueError):
    """A model file that is not torch-weights/1 (an ONNX graph: the gen's variant was not registered)."""


def read(path):
    """The file's dict, checked (`weights_only=True`) -> dict; NotTorchWeights for anything else."""
    with open(path, 'rb') as f:
        head = f.read(4)
    if head[:2] != b'PK':
        raise NotTorchWeights(f'{path}: not a torch-weights/1 file (a torch.save zip archive; an ONNX graph? the gen\'s '
                              f'torch variant may not be registered)')
    try:
        d = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as e:  # noqa: BLE001 - the safe unpickler's refusal, a broken archive
        raise NotTorchWeights(f'{path}: not a torch-weights/1 file ({type(e).__name__}: {str(e)[:300]})') from None
    if not isinstance(d, dict) or d.get('format') != FORMAT:
        raise NotTorchWeights(f'{path}: format {d.get("format") if isinstance(d, dict) else type(d).__name__!r}, not {FORMAT}')
    for k in ('net_config', 'state_dict'):
        if not isinstance(d.get(k), dict):
            raise NotTorchWeights(f'{path}: no {k}')
    return d


def build(d, device='cpu', game=None):
    """A read file's network on `device` (zero.net.load's steps: the encoding guard, the weights, the widened head)."""
    from .net import aux_layout, build as build_net, check_encoding, game_aux
    check_encoding(d['net_config'], game, '<torch-weights>')
    m = build_net(d['net_config'])
    m.load_state_dict(d['state_dict'])
    if d.get('aux_older'):
        m.aux_older = [aux_layout(o) for o in d['aux_older']]
    m = game_aux(m, game)
    return m.to(device).eval()


def load(path, device='cpu', game=None):
    """torch-weights/1 -> (network on `device`, its `infer` settings)."""
    d = read(path)
    inf = dict(infer_of({}))
    inf.update({k: d['infer'][k] for k in INFER_KEYS if k in (d.get('infer') or {})})
    return build(d, device, game), inf
