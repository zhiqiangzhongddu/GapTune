# GapTune forward drivers

The drivers replay the vanilla encoders from `src.model.build_encoder_from_cfg`
(no encoder substitution; checkpoints load strictly) and inject the two
GapTune interventions of paper Eq. 13-16:

- **node prompt** `p^N` `[N, d0]`, added once to the initial continuous state
  `h^(0)` (after any frozen input embedding);
- **message prompts** `p^Ml` `[E_l, d_Ml]`, added to the native message
  *after* structural weighting / attention and *before* aggregation:
  `m~ = a_vu * v_u + p` (EdgePrompt uses `a_vu * f(h_u + p)`; the two differ
  by `(a_vu - 1) p`). Attention is recomputed from the prompted states.

`collect=True` returns `h0`, `h_final` (= node_repr) and per layer the state
fed to that layer (`inputs`), the native messages and their
`sender`/`receiver` rows. Prompt rows align with these rows. Descriptor
dims: `e_N = d0 + d_L`, `e_Ml = 2 * dim(inputs_l) + d_Ml`.

| model | h0 (d0) | message `m_{u->v}` (d_Ml) | self-loop messages | UPD keeps |
|---|---|---|---|---|
| `gcn` | `x` (in) | `norm_vu * W h_u` (out_c) | added (gcn_norm) | bias, act/bn/dropout |
| `gin` | `x` (in) | `w_vu * h_u` (in_c), no relu | input edges only | `(1+eps) h_v`, MLP |
| `gat` | `x` (in) | `alpha_vu * W h_u`, heads x C | remove + re-add | head mean, bias |
| `transformer` | `x` (in) | `alpha_vu * (W_V h_u + b_V)`, heads x C | input edges only | head mean, `lin_skip(h_v)` |
| `fagcn` | `lin_in(x)` (hidden) | `tanh(a_l h_u + a_r h_v) * norm_vu * h_u` (hidden) | added (gcn_norm) | `eps * h~0` (prompted) |
| `h2gcn` | `x` (in) | hop 1: `S_vu x_u` (in); hop 2: `S_vu h1_u` (hidden); sender = col, receiver = row | appended (existing kept) | lin/bn/act, concat |
| `nodeformer` | `act(LN(fcs0(x)))` (hidden) | `alpha_vu * v_u`, heads x hidden, on input edges | input edges only | `u~ + r~ + sum p`, then Wo/residual/LN/act |

`edge_weight` (a multiplicative weight on `data.edge_index`, used only by
source-free inversion) is supported by `gcn` (weighted `gcn_norm`, native
self-loop fill 1) and `gin` (weighted neighbour message); the other drivers
raise `NotImplementedError`.

**NodeFormer.** The vendored conv reseeds its random-feature projection from
`sum(query)`. The driver instead requires fixed per-layer projections from
`src/finetune/encoders/nodeformer_fixed.build_fixed_projections(model, seed)`
(seed `seed + 1009 * l`, `l = 1..L`, paper B.10), reused for source, target
and prompted passes. `alpha_vu` comes from the native kernel features and
per-graph denominators (`kernelized_softmax(..., return_weight=True)`);
the all-pair aggregate and any native relational bias are unchanged.
Zero-prompt parity is defined against the native operator given the same
projections.

**Seed streams.** Minibatch order (paper D.2 stream `s+5`) has no dedicated
generator: the shared runner draws it from the global seed `s` (finetuner
`set_seed` + unseeded train-loader shuffle), identically across GapTune arms
(prompt options, GapTune/GapTune+, proxy mode and budget).
