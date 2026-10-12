"""pytest setup: import zero from this checkout (python/zero/_engine.pyd: tools/build_py.sh)."""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

torch.set_num_threads(1)          # docs/az.md 7: > 1 thread collapses on a saturated CPU
