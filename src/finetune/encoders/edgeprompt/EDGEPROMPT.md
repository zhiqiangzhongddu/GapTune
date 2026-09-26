# EdgePrompt backbone coverage

EdgePrompt injects a learnable per-edge prompt into the message path of the
pretrained encoder. Every supported backbone defines exactly where the prompt
enters its native update; `mlp` is not a GNN and is not supported.
In every backbone the prompt is added to the sender state before structural
weighting (`a_ji * f(h_j + p_ji)`), never to the already-weighted message.

| model | status | injection formula summary |
|---|---|---|
| `gcn` | official | normalized `h_j + p_ji` message |
| `gin` | official | summed `h_j + p_ji` message, then MLP |
| `gat` | extension | attention from unprompted `h`; value = `W(h_j + p_ji)` |
| `transformer` | extension | value = `W_V(h_j + p_ji)`; query/key unchanged |
| `h2gcn` | extension | prompted per-hop aggregation; lin/act/dropout/concat preserved |
| `fagcn` | extension | gate from unprompted features; message on `(h_j + p_ji)` |
| `nodeformer` | extension | prompted relational-bias path (requires `model.nodeformer.rb_order >= 1`) |
| `mlp` | unsupported | not a GNN |

Per-backbone implementations are sibling files in this directory
(`gcn_gin.py`, `gat.py`, `transformer.py`, `h2gcn.py`, `fagcn.py`,
`nodeformer.py`). Each module advertises
`edgeprompt_support` and `edgeprompt_formula` attributes for
introspection.

## NodeFormer note

EdgePrompt on NodeFormer injects into the native relational-bias path,
which only exists when `model.nodeformer.rb_order >= 1`. The default
`model.nodeformer.rb_order` in `src/config/_model.py` is set so that
NodeFormer pretraining produces compatible checkpoints out of the box.
Any NodeFormer pretrain produced with `rb_order=0` cannot be used with
EdgePrompt and must be re-pretrained.
