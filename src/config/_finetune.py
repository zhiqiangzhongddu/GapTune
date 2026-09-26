from yacs.config import CfgNode as CN

from ._dataset import _default_dataset_cfg


def set_finetune_cfg(cfg: CN) -> None:
    """Attach finetune config defaults to *cfg*."""
    cfg.finetune = CN()
    cfg.finetune.method = "supervised"  # supervised, all_in_one, edgeprompt, gaptune, gpf, gppt, graphprompt, igap, mtg, pronog, supt
    cfg.finetune.num_runs = 5  # number of finetuning runs with different seeds
    # finetune dataset options
    cfg.finetune.dataset = _default_dataset_cfg()
    cfg.finetune.dataset.fixed_split = (0.1, 0.1, 0.8)  # train, val, test split ratios
    # general finetuning options
    cfg.finetune.epochs = 200  # maximum number of finetuning epochs
    cfg.finetune.early_stopping = 20  # early stopping patience (0 to disable)
    cfg.finetune.lr = 1e-3  # learning rate
    cfg.finetune.weight_decay = 0.0  # weight decay
    cfg.finetune.batch_size = 32  # used for induced tasks and graph-level tasks
    cfg.finetune.num_workers = 0  # number of data loading workers
    cfg.finetune.checkpoint_dir = "outputs/finetuned_models"  # directory to save checkpoints
    cfg.finetune.log_dir = "outputs/logs/finetuning_models"  # directory to save finetune logs
    cfg.finetune.monitor_metric = "auto"  # finetune monitor setting; auto is resolved by the shared monitoring policy
    cfg.finetune.edge_readout = "pool"  # readout on induced edge tasks for methods routed through TaskAwareObjective: pool (legacy whole-subgraph pooling) or endpoints (Hadamard of the two endpoint reps, the canonical LP readout). pool preserves the run identity of all published rows.
    cfg.finetune.grad_clip = 0.0  # max gradient norm (0.0 = disabled); applied by methods whose training loop supports clipping, including supervised, gpf and edgeprompt
    cfg.finetune.normalize_regression_targets = True  # shared train-split, per-target z-scoring for every regression finetune method; predictions/metrics stay on the original scale
    cfg.finetune.regression_loss = "normalized_mse"  # normalized_mse (legacy) or metric_mae (train-std-weighted normalized L1, proportional to original-unit flattened MAE)
    cfg.finetune.multilabel_loss = "masked_bce"  # masked_bce (legacy) or train-only per-target/per-class macro_balanced_bce
    cfg.finetune.skip_if_exists = True  # skip finetune run if checkpoint already exists
    cfg.finetune.run_tasks_tsv = False  # when True, iterate through available datasets/tasks defined in tasks_tsv; set False to use finetune.dataset settings
    cfg.finetune.tasks_tsv = "slurm/finetune.tsv"  # whitespace-delimited finetune rows: dataset task_level induced [task_type] [fixed_split] [pretrained_run_name] [finetune_method]
    cfg.finetune.pretrained_checkpoint = ""  # explicit checkpoint path override (optional)
    cfg.finetune.frozen_load_min_match_ratio = 1.0  # when the method requires a frozen encoder, fraction of the encoder's ``pretrained_key_whitelist`` that must load from the pretrained checkpoint; 1.0 = strict. Prompt-only keys are never counted. Applies to every frozen-encoder finetune method.
    # supervised finetuning method-specific options
    cfg.finetune.supervised = CN()
    cfg.finetune.supervised.freeze_encoder = False  # official pretrain-gnns finetunes encoder end-to-end
    # all_in_one finetuning method-specific options
    cfg.finetune.all_in_one = CN()
    cfg.finetune.all_in_one.freeze_encoder = True  # All-in-one tunes prompts only; encoder must be frozen
    cfg.finetune.all_in_one.token_num = 10  # number of prompt tokens
    cfg.finetune.all_in_one.cross_prune = 0.1  # threshold for prompt->graph cross edges
    cfg.finetune.all_in_one.inner_prune = 0.3  # threshold for prompt inner edges
    cfg.finetune.all_in_one.total_epochs = 1000  # official-style total epoch budget for all_in_one (<=0 uses finetune.epochs)
    cfg.finetune.all_in_one.cache_answer_embeddings = True  # speed-up: cache frozen-encoder embeddings during answering phase
    cfg.finetune.all_in_one.answer_with_softmax = False  # optional strict parity mode: reference head is Linear+Softmax with CE
    cfg.finetune.all_in_one.bidirectional_cross_edges = None  # None auto-selects by task (node-induced=True, graph=False); set bool to force behavior
    cfg.finetune.all_in_one.exclude_prompt_from_pooling = None  # None auto-selects by task (node-induced=True, graph=False); set bool to force behavior
    cfg.finetune.all_in_one.answer_epoch = -1  # <=0 uses default auto schedule: node-induced/few-shot=50, graph-standard=5
    cfg.finetune.all_in_one.prompt_epoch = -1  # <=0 uses default auto schedule: node-induced/few-shot=50, graph-standard=1
    cfg.finetune.all_in_one.prompt_lr = 1e-6  # prompt optimizer learning rate
    cfg.finetune.all_in_one.prompt_weight_decay = None  # None falls back to finetune.weight_decay (reference CLI default: 0)
    cfg.finetune.all_in_one.answer_lr = 1e-3  # answering head optimizer learning rate
    cfg.finetune.all_in_one.answer_weight_decay = None  # None falls back to finetune.weight_decay (reference CLI default: 0)
    # edgeprompt finetuning method-specific options
    cfg.finetune.edgeprompt = CN()
    cfg.finetune.edgeprompt.freeze_encoder = True  # EdgePrompt tunes prompts only; encoder must be frozen
    cfg.finetune.edgeprompt.plus = True  # prefer anchor-conditioned prompts; set False to recover vanilla EdgePrompt
    cfg.finetune.edgeprompt.num_anchors = None  # None uses baseline-style defaults: 10 for node tasks, 5 for graph tasks
    cfg.finetune.edgeprompt.lr = None  # None falls back to finetune.lr at runtime
    cfg.finetune.edgeprompt.weight_decay = None  # None falls back to finetune.weight_decay at runtime
    cfg.finetune.edgeprompt.force_mean_pooling = True  # use mean readout to match the released EdgePrompt downstream setup
    cfg.finetune.edgeprompt.pin_bn_eval_when_frozen = False  # keep BN in train mode unless strict frozen-BN behavior is required
    cfg.finetune.edgeprompt.gin_message_relu = True  # keep GIN message nonlinearity aligned with the baseline operator
    cfg.finetune.edgeprompt.gin_train_eps = True  # allow eps to adapt during prompt tuning for baseline-like GIN behavior
    cfg.finetune.edgeprompt.use_official_node_subgraphs = True  # use 2-hop node-induced subgraphs without aggressive size clipping
    cfg.finetune.edgeprompt.node_subgraph_hops = 2  # official node downstream extracts fixed 2-hop subgraphs
    cfg.finetune.edgeprompt.node_subgraph_min_size = 1  # avoid hop-expansion by minimum-size heuristics
    cfg.finetune.edgeprompt.node_subgraph_max_size = 100000  # keep almost all sampled nodes to avoid random truncation
    # gpf finetuning method-specific options
    cfg.finetune.gpf = CN()
    cfg.finetune.gpf.freeze_encoder = True  # GPF tunes prompts only; encoder must be frozen
    cfg.finetune.gpf.plus = False  # when True use GPF+, otherwise GPF
    cfg.finetune.gpf.p_num = 5  # number of prompt bases for GPF+
    cfg.finetune.gpf.lr = None  # None falls back to finetune.lr at runtime
    cfg.finetune.gpf.weight_decay = None  # None falls back to finetune.weight_decay at runtime
    cfg.finetune.gpf.prompt_lr = None  # None falls back to gpf.lr (then finetune.lr) at runtime
    cfg.finetune.gpf.prompt_weight_decay = None  # None falls back to gpf.weight_decay (then finetune.weight_decay) at runtime
    cfg.finetune.gpf.head_lr_scale = 1.0  # scale applied to base lr for prediction head
    cfg.finetune.gpf.head_lr = None  # None falls back to gpf.lr * head_lr_scale at runtime
    cfg.finetune.gpf.head_weight_decay = None  # None falls back to gpf.weight_decay (then finetune.weight_decay) at runtime
    cfg.finetune.gpf.head_layers = 1  # official GPF head depth knob (`num_layers` in upstream scripts)
    cfg.finetune.gpf.head_hidden_dim = 0  # <=0 uses input representation dim for hidden layers
    cfg.finetune.gpf.head_dropout = 0.0  # optional dropout between MLP head layers
    cfg.finetune.gpf.freeze_encoder_bn_when_frozen = True  # keep frozen encoder BN stats fixed during prompt tuning
    cfg.finetune.gpf.prefer_non_induced_node = True  # intentional: node GPF runs on the full graph with masks
    cfg.finetune.gpf.monitor_train_loss = False  # when True, auto monitor train_loss even when val split exists
    # supt finetuning method-specific options
    cfg.finetune.supt = CN()
    cfg.finetune.supt.freeze_encoder = True  # SUPT tunes prompts only; encoder must be frozen
    cfg.finetune.supt.variant = "soft"  # soft (official DiffPoolPrompt: per-node softmax mixture of bases) or hard (official SAGPoolPrompt: per-graph top-k nodes per basis)
    cfg.finetune.supt.num_bases = 5  # number of prompt bases k (official README commands use 5)
    cfg.finetune.supt.ratio = 0.4  # hard only: per-graph selection ratio r in (0, 1); ceil(r * N_g) nodes per basis
    cfg.finetune.supt.hard_score = "tanh"  # hard only: tanh (official default) or graph_softmax (official --softmax: per-graph softmax over nodes)
    cfg.finetune.supt.orth_loss = False  # when True (and num_bases > 1) orthogonal basis init plus ||B B^T - I||_F added to the loss
    cfg.finetune.supt.gcn_bias = True  # bias of the GCNConv scorer (official code; paper Eq. 5 omits it)
    cfg.finetune.supt.head_layers = 1  # prediction head depth, built like GPF's head
    cfg.finetune.supt.head_hidden_dim = 0  # <=0 uses input representation dim for hidden layers
    cfg.finetune.supt.head_dropout = 0.0  # optional dropout between MLP head layers
    cfg.finetune.supt.lr = None  # None falls back to finetune.lr at runtime
    cfg.finetune.supt.weight_decay = None  # None falls back to finetune.weight_decay at runtime
    cfg.finetune.supt.prompt_lr = None  # None falls back to supt.lr (then finetune.lr) at runtime
    cfg.finetune.supt.prompt_weight_decay = None  # None falls back to supt.weight_decay (then finetune.weight_decay) at runtime
    cfg.finetune.supt.head_lr = None  # None falls back to supt.lr (then finetune.lr) at runtime
    cfg.finetune.supt.head_weight_decay = None  # None falls back to supt.weight_decay (then finetune.weight_decay) at runtime
    cfg.finetune.supt.freeze_encoder_bn_when_frozen = True  # keep frozen encoder BN stats fixed during prompt tuning (as GPF)
    # igap finetuning method-specific options
    cfg.finetune.igap = CN()
    cfg.finetune.igap.freeze_encoder = True  # IGAP tunes prompts + head only; encoder must be frozen
    cfg.finetune.igap.num_signal_prompts = 16  # L graph-signal prompt bases P_s (paper default)
    cfg.finetune.igap.num_eigvecs = 32  # K lowest-frequency Laplacian eigenvectors aligned by P_t (paper default)
    cfg.finetune.igap.tau = 0.1  # temperature of the cosine label-prototype logits (P_l)
    cfg.finetune.igap.use_signal_prompt = True  # False = paper ablation "No P_s"
    cfg.finetune.igap.use_spectral_prompt = True  # False = paper ablation "No P_t"
    cfg.finetune.igap.use_label_prompt = True  # False = paper ablation "No P_l, end2end" (always off for regression)
    cfg.finetune.igap.head_hidden_dim = 0  # hidden width of the 2-layer ReLU head; <=0 uses model.out_dim
    cfg.finetune.igap.lr = None  # None falls back to finetune.lr at runtime
    cfg.finetune.igap.weight_decay = None  # None falls back to finetune.weight_decay at runtime
    # mtg finetuning method-specific options
    cfg.finetune.mtg = CN()
    cfg.finetune.mtg.freeze_encoder = True  # MTG tunes message prototypes + head only; encoder must be frozen
    cfg.finetune.mtg.num_prototypes = 10  # m message prototypes per layer (official default)
    cfg.finetune.mtg.lr = None  # None falls back to finetune.lr at runtime
    cfg.finetune.mtg.weight_decay = None  # None falls back to finetune.weight_decay at runtime
    # gaptune finetuning method-specific options (paper Table 17; ablations App. C)
    cfg.finetune.gaptune = CN()
    cfg.finetune.gaptune.freeze_encoder = True  # GapTune tunes prompts + head only; encoder must be frozen
    cfg.finetune.gaptune.plus = True  # True: GapTune+ (pretraining graphs as source); False: GapTune (source-free proxy graphs)
    cfg.finetune.gaptune.num_queries = 8  # K queries per observation type
    cfg.finetune.gaptune.tau_c = 0.5  # context-pooling temperature (Eq. 8)
    cfg.finetune.gaptune.tau_p = 0.5  # local-relevance temperature (Eq. 11)
    cfg.finetune.gaptune.obs_eps = 1e-6  # eps of the observation normalization nu(z) = z / sqrt(||z||^2 + eps^2)
    cfg.finetune.gaptune.value_mode = "gap"  # gap, target, source, paired_mean, or free (learned values replacing the queries)
    cfg.finetune.gaptune.prompt_locations = "node_message"  # node_message, node, message, or none (head-only control, same readout)
    cfg.finetune.gaptune.query_mode = "shared"  # shared, frozen (queries not trained), or untied (separate source/target queries)
    cfg.finetune.gaptune.mixture = "local"  # local (Eq. 11), uniform (1/K), or global (learned logits shared per type)
    cfg.finetune.gaptune.gate = "signed"  # signed (tanh) or nonnegative (theta projected onto [0, inf) after every step)
    cfg.finetune.gaptune.source_max_nodes = 512  # fixed source sample: node observations
    cfg.finetune.gaptune.source_max_messages = 2048  # fixed source sample: directed messages per layer
    cfg.finetune.gaptune.include_self_loop_messages = True  # False drops self-loop messages from pooling and prompting
    cfg.finetune.gaptune.lr = None  # None falls back to finetune.lr at runtime
    cfg.finetune.gaptune.weight_decay = None  # None falls back to finetune.weight_decay at runtime
    cfg.finetune.gaptune.grad_clip = 5.0  # global norm clip of the trainable gradients
    # source-free proxy construction for plus=False (paper App. B.8, Table 18)
    cfg.finetune.gaptune.proxy = CN()
    cfg.finetune.gaptune.proxy.mode = "inverted"  # inverted (pretext-optimized proxies) or random (same init, no optimization)
    cfg.finetune.gaptune.proxy.num_graphs = 16  # B proxy graphs
    cfg.finetune.gaptune.proxy.num_nodes = 32  # nodes per proxy graph
    cfg.finetune.gaptune.proxy.updates = 1000  # joint inversion updates (last iterate kept)
    cfg.finetune.gaptune.proxy.lr = 1e-2  # Adam learning rate of the inversion
    cfg.finetune.gaptune.proxy.grad_clip = 5.0  # gradient-norm clip of the inversion
    cfg.finetune.gaptune.proxy.lambda_x = 1e-4  # feature regularizer weight
    cfg.finetune.gaptune.proxy.lambda_a = 1.0  # density regularizer weight
    cfg.finetune.gaptune.proxy.density = 4 / 31  # rho, density prior (output bias log(rho / (1 - rho)))
    cfg.finetune.gaptune.proxy.final_edges = 64  # undirected edges kept per proxy graph
    cfg.finetune.gaptune.proxy.edge_hidden = 128  # hidden width of the shared edge MLP
    cfg.finetune.gaptune.proxy.feature_radius = 10.0  # proxy feature rows projected onto this ball
    cfg.finetune.gaptune.proxy.tau_start = 1.0  # relaxation temperature at update 0
    cfg.finetune.gaptune.proxy.tau_end = 0.1  # relaxation temperature at the last update (geometric schedule)
    cfg.finetune.gaptune.proxy.edgepred_pos_pairs = 32  # EdgePred positive conditioning pairs per proxy
    cfg.finetune.gaptune.proxy.edgepred_neg_pairs = 32  # EdgePred negative conditioning pairs per proxy
    cfg.finetune.gaptune.proxy.edgepred_logit = 2.1972245773362196  # log(9), fixed conditioning-pair logit magnitude
    cfg.finetune.gaptune.proxy.graphcl_tau = 0.2  # GraphCL NT-Xent temperature
    cfg.finetune.gaptune.proxy.graphcl_edge_drop = 0.1  # GraphCL undirected edge-drop probability
    cfg.finetune.gaptune.proxy.graphcl_feature_mask = 0.1  # GraphCL elementwise feature-mask probability
    # gppt finetuning method-specific options
    cfg.finetune.gppt = CN()
    cfg.finetune.gppt.freeze_encoder = True  # GPPT tunes prompts only; encoder must be frozen
    cfg.finetune.gppt.center_num = 0  # number of structure centers (0 -> auto: num_classes)
    cfg.finetune.gppt.lr = 2e-3  # official GPPT prompt optimizer learning rate
    cfg.finetune.gppt.weight_decay = 5e-4  # official GPPT prompt optimizer weight decay
    cfg.finetune.gppt.constraint_weight = 1e-2  # orthogonality regularization on task tokens (matches original GPPT lr_c=0.01)
    cfg.finetune.gppt.structure_mode = "concat"  # feature mode for StructureToken: node, neighbor, concat
    cfg.finetune.gppt.task_mode = "concat"  # feature mode for TaskToken heads: node, neighbor, concat
    cfg.finetune.gppt.add_self_loops = False  # add self-loops in mean-neighbor aggregation
    cfg.finetune.gppt.update_structure_every_step = True  # refresh structure token centers after each step
    cfg.finetune.gppt.update_structure_from_mask = False  # official: re-cluster on full mid_h; set True to restrict to labeled nodes
    cfg.finetune.gppt.kmeans_max_iter = 100  # max kmeans iterations for center updates
    cfg.finetune.gppt.kmeans_tol = 1e-4  # convergence tolerance for center updates
    cfg.finetune.gppt.kmeans_restarts = 1  # number of kmeans restarts per update
    cfg.finetune.gppt.use_sklearn_kmeans = True  # align with official implementation when sklearn is available
    cfg.finetune.gppt.kmeans_random_state = 0  # random_state for sklearn KMeans
    cfg.finetune.gppt.kmeans_n_init = 10  # n_init for sklearn KMeans
    # pronog finetuning method-specific options
    cfg.finetune.pronog = CN()
    cfg.finetune.pronog.freeze_encoder = True  # ProNoG tunes the condition-net only; encoder must be frozen
    cfg.finetune.pronog.hops = 2  # ego-network radius (delta) for the conditioning readout; paper uses 2
    cfg.finetune.pronog.neighbor_cap = 20  # per-hop cap on ego-network members (official k=20); 0 = uncapped
    cfg.finetune.pronog.bottleneck_dim = 64  # condition-net bottleneck width m (paper: 64)
    cfg.finetune.pronog.condition_dropout = 0.1  # dropout on the condition-net input (official PromptVector)
    cfg.finetune.pronog.condition_scaling = 0.1  # scaling on generated prompts (official PromptVector); 0 zeroes the prompts (no-prompt ablation, add-combine only)
    cfg.finetune.pronog.prompt_combine = "add"  # add (official code: h + p) or mul (paper Eq. 9: element-wise p * h)
    cfg.finetune.pronog.tau = 1.0  # temperature (> 0) for prototype cosine logits (official downstream applies none)
    cfg.finetune.pronog.graph_pooling = "sum"  # sum/add (paper Eq. 10 readout), mean, max, or target
    cfg.finetune.pronog.train_center_mode = "batch"  # batch (official: prototypes from the current train batch, gradients flow; missing classes filled from the previous epoch's bank) or train (detached epoch-level train-split centers)
    cfg.finetune.pronog.eval_center_mode = "train"  # train only; batch would leak evaluation labels
    cfg.finetune.pronog.prompt_lr = 1e-4  # Adam lr for the condition-net (official down_lr)
    cfg.finetune.pronog.prompt_weight_decay = 0.0  # official downstream Adam runs without weight decay
    # graphprompt finetuning method-specific options
    cfg.finetune.graphprompt = CN()
    cfg.finetune.graphprompt.freeze_encoder = True  # GraphPrompt tunes prompts only; encoder must be frozen
    cfg.finetune.graphprompt.plus = False  # when True use GraphPrompt+ stage-wise prompt parameterization
    cfg.finetune.graphprompt.p_num = 4  # number of prompt masks for official GraphPrompt+
    cfg.finetune.graphprompt.init = "xavier"  # xavier or identity
    cfg.finetune.graphprompt.init_std = 0.02  # std for identity init noise
    cfg.finetune.graphprompt.tau = 0.1  # temperature for prototype contrastive classification
    cfg.finetune.graphprompt.score_mode = "neg_distance"  # neg_distance, distance, or cosine
    cfg.finetune.graphprompt.loss_reduction = "mean"  # mean or sum
    cfg.finetune.graphprompt.train_center_mode = "ema"  # batch, train, or ema; official uses batch-only, ema is more stable
    cfg.finetune.graphprompt.eval_center_mode = "train"  # train or batch
    cfg.finetune.graphprompt.graph_pooling = "sum"  # encoder, sum/add, mean, max, or target
    cfg.finetune.graphprompt.prompt_dropout = 0.0  # dropout on prompted embeddings; original GraphPrompt uses 0.5
    cfg.finetune.graphprompt.center_momentum = 0.9  # EMA momentum for prototype bank when train_center_mode=ema
    cfg.finetune.graphprompt.prompt_lr = 1e-3  # prompt optimizer learning rate (official node downstream uses 0.1)
    cfg.finetune.graphprompt.prompt_weight_decay = 1e-5  # official GraphPrompt default weight decay
    cfg.finetune.graphprompt.repr_source = "last"  # "last" (final layer) or "layer_concat" (all layers; official GraphPrompt uses layer_concat)
    # Node-level embedding postprocessing (official GraphPrompt node downstream only)
    cfg.finetune.graphprompt.embedding_postprocess = "none"  # "none" or "official_node" (sigmoid + adj propagation)
    cfg.finetune.graphprompt.nhop_neighbour = 1  # adjacency propagation hops when embedding_postprocess=official_node
    cfg.finetune.graphprompt.self_loop_weight = 1.0  # self-loop weight added to adjacency for propagation
