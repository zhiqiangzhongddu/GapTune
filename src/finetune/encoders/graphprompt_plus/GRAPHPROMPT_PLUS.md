# GraphPrompt+ backbone coverage

GraphPrompt+ injects a stack of learnable elementwise prompt masks
("stage prompts") at four points in the encoder forward pass:

| stage | injection point |
|---|---|
| 0 | input node features (before layer 0) |
| 1 | between layer 0 and layer 1 (after act/bn/dropout) |
| 2 | between layer 1 and layer 2 (after act/bn/dropout) |
| 3 | final node representation (before graph pooling) |

Stages 1 and 2 only activate when the encoder has enough layers
(`num_layers >= 2` and `num_layers >= 3` respectively).  Each stage
produces one branch; branches are combined by softmaxed learnable
weights into a single embedding.

The driver of this forward pass is per-backbone: each
`GraphPromptPlusAdapter` knows how to walk through its backbone's
forward pass and inject `prompt.apply_stage(stage_id, x)` at the right
point.  Phase 1 ships a single shared adapter for the GNN-stack
backbones; subsequent phases land per-backbone adapters for the
non-stack encoders.

## Coverage table

| model | status | adapter | notes |
|---|---|---|---|
| `gcn`         | official  | `GNNStackAdapter`         | uniform `conv → act → bn → dropout` stack |
| `gin`         | official  | `GNNStackAdapter`         | uniform `conv → act → bn → dropout` stack |
| `gat`         | extension | `GNNStackAdapter`         | same stack; stage-wise prompt was not in the original GraphPrompt+ paper for GAT |
| `mlp`         | extension | `GNNStackAdapter`         | linear layers; prompt is purely elementwise |
| `transformer` | extension | `TransformerStackAdapter` | TransformerConv stack; no BN; **no layer_concat** |
| `nodeformer`  | extension | `NodeFormerStackAdapter`  | input proj → conv stack (with residual + bn + dropout) → output proj; **no layer_concat** |
| `fagcn`       | extension | `FAGCNStackAdapter`       | between-layer prompt on `x` only; FAConv residual `x_0` left unprompted (see file docstring); **no layer_concat** |
| `h2gcn`       | extension | `H2GCNAdapter`            | fixed 2-hop; only stages **{0, 1, 3}** active; stage 1 prompts `x_1` and flows to both the 2-hop aggregation and the final concat |

## Adapter contract

Every adapter declares:

- `support: str` — `"official"` for paper-faithful injection points,
  `"extension"` when the repo introduces a design choice (the choice
  must be documented in the adapter's module docstring).
- `formula: str` — short human-readable description; surfaces in error
  messages.

And implements four classmethods:

- `supports_model(model)` — runtime guard that the supplied encoder
  exposes the API the adapter needs.
- `iter_stage_specs(num_layers, in_dim, hidden_dim, out_dim, repr_dim)`
  — list of `(stage_id, dim)` slots this backbone exposes; the prompt
  module sizes its masks from this list.
- `supports_layer_concat()` — whether `repr_source="layer_concat"` is
  meaningful for this backbone.
- `forward_with_stage_prompt(model, data, stage_id, prompt, repr_source)`
  — drives one stage-prompted forward pass and returns
  `(node_repr, graph_repr_or_None)`.  Node masking, label preparation,
  graph pooling fallback, and cross-stage mixing live in the
  `FinetuneGraphPrompt` task class — adapters return raw stage
  representations only.

## Why this lives under `encoders/`

`src/finetune/encoders/` is the home for per-backbone forward-pass
modifications used by finetune methods.  GraphPrompt+ on `gcn/gin/gat/
mlp` does not require an encoder *subclass* (the standard `GNNEncoder`
exposes `convs`/`bns`/`act`/`dropout` so an external driver can replay
the forward), but later phases (transformer / nodeformer / h2gcn /
fagcn) likely will.  Keeping all GraphPrompt+ adapters in one place
matches the EdgePrompt layout and avoids splitting the package between
two directories when the harder backbones land.
