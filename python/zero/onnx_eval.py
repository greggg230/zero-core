"""`OnnxEvaluator`: a `zero-onnx/1` graph (zero/onnx_export.py, docs/az.md 41) answering the driver's network
batches through onnxruntime on the CPU, with the generic step every zero-core evaluator does - a drop-in
for `infer.NetEvaluator` wherever a LeafBatch evaluator is taken (`infer.drive`, zero.h2h's seats, a remote
worker), and the Python reference of the native worker's evaluator (zero-farm parity gate 2, docs/az.md 44).

    ev = OnnxEvaluator('eoe1_iter_0080.packed.onnx', threads=1)
    drive(driver, {'m': ev})

A batch's rows go to the graph as CSR (`x_ptr` / `x_idx` / `x_val`; a dense batch's non-zeros are taken),
batches over `chunk` rows in pieces of at most `chunk` rows grouped by non-zero count
(`onnx_parity.run_chunked`), then `onnx_parity.answer`: each row's legal logits softmaxed per segment in
float32 (`onnx_parity.exact_softmax`: the native worker's bits), value and margin as the graph gives them, the root belief rows' auxiliary
outputs (a graph without the head refuses such a batch: net.HeadMissing). Session: CPU EP, `threads`
intra-op threads, inter-op 1, sequential, all graph optimizations (the native worker's). `expect`
checks the graph's metadata as the native worker does (game, features, actions).
"""
import numpy as np

from .onnx_parity import Rows, answer, run_chunked, session
from .protocol import ROW_BELIEF, belief_rows


class OnnxEvaluator:
    def __init__(self, path, threads=1, chunk=64, expect=None):
        self.path = str(path)
        self.sess = session(self.path, threads)
        self.meta = dict(self.sess.get_modelmeta().custom_metadata_map)
        if self.meta.get('zero_onnx') != '1':
            raise ValueError(f'{path}: zero_onnx {self.meta.get("zero_onnx")!r}: not a zero-onnx/1 graph')
        for k, v in (expect or {}).items():
            if str(self.meta.get(k)) != str(v):
                raise ValueError(f'{path}: metadata {k} {self.meta.get(k)!r}, expected {v!r}')
        self.features = int(self.meta['features'])
        self.aux_width = int(self.meta['aux_width'])
        self.chunk = int(chunk)
        self.calls = 0
        self.leaves = 0

    def rows(self, lb):
        """The batch as `onnx_parity.Rows` (its sparse rows; a dense batch's non-zeros)."""
        a = lb.a
        extra = a.get('extra')
        extra = np.zeros(lb.B, np.uint8) if extra is None else extra
        if 'xs_val' in a:
            if int(a['n_features']) != self.features:
                from .net import EncodingMismatch
                raise EncodingMismatch(f'a sparse batch of {int(a["n_features"]):,} features per row for a graph of '
                                       f'{self.features:,}')
            return Rows(a['xs_off'], a['xs_idx'], a['xs_val'], a['n_players'], a['opt_off'], a['opt_action'], extra,
                        self.features)
        x = np.asarray(a['x'], np.float32)
        if x.shape[1] != self.features:
            from .net import EncodingMismatch
            raise EncodingMismatch(f'a batch of {x.shape[1]:,} features per row for a graph of {self.features:,}')
        return Rows.from_dense(x, a['n_players'], a['opt_off'], a['opt_action'], extra)

    def answer_rows(self, rows):
        if len(belief_rows(rows.extra)) and not self.aux_width:
            from .net import HeadMissing
            raise HeadMissing(f'{self.path}: a batch with root belief rows for a graph without the auxiliary head')
        out = run_chunked(self.sess, rows, self.chunk)
        self.calls += 1
        self.leaves += rows.B
        return answer(out, rows)

    def __call__(self, lb):
        return self.answer_rows(self.rows(lb))

    def evaluate(self, x, n_players, opt_off, opt_action, extra=None):
        """NetEvaluator.evaluate's signature (dense rows)."""
        extra = np.zeros(len(x), np.uint8) if extra is None else np.asarray(extra, np.uint8)
        return self.answer_rows(Rows.from_dense(np.asarray(x, np.float32), n_players, opt_off, opt_action, extra))

    def launch(self, lb):                          # (infer.drive's pipelining: computed at once)
        return self(lb)

    def result(self, h):
        return h


__all__ = ['OnnxEvaluator', 'ROW_BELIEF']
