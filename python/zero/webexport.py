"""Checkpoint -> `.bin` weight file for the browser players (crate `zero_webnet`, which reads
it; moved from Caverna Zero's tools/export_web.py, whose wasm player uses it).

Format (little endian): b"CVZW", u32 version 1, u32 header length, the JSON header (features,
actions, hidden, blocks, head, seats, game, source, iter, tensors = [{name, shape, offset (f16
elements)}]), spaces to an even length, the float16 data. The margin head is not exported
(unused by the browser search), nor are the auxiliary heads (training only).

  python -m zero.webexport runs/x/models/iter_0040.pt out.bin
"""
import json
import os
import struct
import sys
import threading

import numpy as np


def export_model(ckpt, out):
    """Checkpoint -> .bin weight file. Returns (params, max relative f16 rounding error)."""
    from .net import load
    model, extra = load(ckpt, 'cpu')
    cfg = model.config
    tensors, chunks, off, worst = [], [], 0, 0.0
    for name, t in model.state_dict().items():
        if name.startswith('mar.') or name.startswith('aux_'):
            continue
        a32 = t.detach().cpu().numpy().astype(np.float32)
        a = a32.astype('<f2')
        if not np.all(np.isfinite(a)):
            raise ValueError(f'{ckpt}: {name} overflows float16')
        big = np.abs(a32) > 1e-3
        if big.any():
            worst = max(worst, float(np.max(np.abs(a.astype(np.float32)[big] - a32[big]) / np.abs(a32[big]))))
        tensors.append({'name': name, 'shape': list(a.shape), 'offset': off})
        chunks.append(a.tobytes())
        off += a.size
    run = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(ckpt))))
    header = {'features': cfg['features'], 'actions': cfg['actions'], 'hidden': cfg['hidden'], 'blocks': cfg['blocks'],
              'head': cfg.get('head', 128), 'seats': cfg['seats'], 'game': cfg.get('game'),
              'source': f'{run}/{os.path.basename(ckpt)}', 'iter': extra.get('iter'), 'tensors': tensors}
    hb = json.dumps(header, separators=(',', ':')).encode('utf-8')
    if (12 + len(hb)) % 2:
        hb += b' '
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = f'{out}.{os.getpid()}.{threading.get_ident()}.tmp'
    with open(tmp, 'wb') as f:
        f.write(b'CVZW' + struct.pack('<II', 1, len(hb)) + hb + b''.join(chunks))
    os.replace(tmp, out)
    return off, worst


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) != 2:
        print(__doc__)
        return 2
    n, worst = export_model(argv[0], argv[1])
    print(f'{argv[1]}: {n} params, {os.path.getsize(argv[1]) / 1e6:.2f} MB, max f16 rounding {worst:.2e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
