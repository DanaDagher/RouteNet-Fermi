# The Two Doors of `traffic` / `packets` and the Locus of Learning in RouteNet-Fermi

**Purpose.** This short report answers three questions that came up while
interpreting the Step 7 pilot MAPE table:

1. **Can we really not remove `traffic` and `packets` from the model?**
   (Yes — we show it from the code and the equation.)
2. **How can we still demonstrate their importance?**
   (Yes — with an inference-time **perturbation** experiment that complements
   the ablation.)
3. **Where is the machine learning actually happening in a "delay
   prediction"?** (The ML predicts **queue occupancy**; the delay formula is
   Little's Law + transmission physics wrapped around it.)

All claims below are grounded in the actual code on branch `xai-protocol-b`
and in `results/delay_decomposition.json`.

---

## 1. Why `traffic` and `packets` cannot be removed from the model

### 1.1 What the data loader will and will not drop

File: [`traffic_models/delay/data_generator.py`](../traffic_models/delay/data_generator.py) — lines 32–35 hard-code the
list of features that the loader is *allowed* to drop:

```python
_DROPPABLE_PATH_SCALARS = frozenset([
    'eq_lambda', 'avg_pkts_lambda', 'exp_max_factor', 'pkts_lambda_on',
    'avg_t_off', 'avg_t_on', 'ar_a', 'sigma',
])
```

`traffic` and `packets` are **deliberately excluded** from this set. Line 59
then does:

```python
dropped = set(_df) & _DROPPABLE_PATH_SCALARS
```

so any request to drop `traffic` or `packets` is silently intersected away and
the loader keeps them in the yielded input dictionary — regardless of what the
variant config says.

### 1.2 What the model does with them on every forward pass

File: [`delay_model.py`](../delay_model.py) — the forward pass always reads them (lines 100–101):

```python
traffic = inputs['traffic']
packets = inputs['packets']
```

and immediately uses them in two **fixed** (non-learned) computations:

```python
# link load (paper eq. 7): drives the link_embedding
load = tf.reduce_sum(tf.gather(traffic, path_to_link[:,:,0]), axis=1) / capacity   # line 120

# per-flow packet size, needed for transmission delay
pkt_size = traffic / packets                                                       # line 122
...
trans_delay = pkt_size * tf.reduce_sum(1 / capacity_gather, axis=1)                # line 192
```

### 1.3 The arithmetic argument (why zero-masking is not a fix)

`pkt_size = traffic / packets`.

- Setting `packets = 0` → division by zero → NaN → the model diverges.
- Setting `traffic = 0` → `trans_delay = 0` for every flow and `load = 0`
  for every link → the model is broken, not tested. Any subsequent MAPE
  number is meaningless.

### 1.4 The two doors, in one picture

**Figure 1** below (draw.io / mermaid) shows the two independent paths through
which `traffic` and `packets` enter the model.

- **Door 1 — the learned embedding** (`path_embedding`, `kept_path_scalars`).
  This *is* controllable per variant. Removing `traffic`/`packets` here means
  the MLP can no longer learn a non-linear pattern *from their normalised
  values*.
- **Door 2 — the fixed physics computations** (`load`, `pkt_size`,
  `trans_delay`). This is **not** controllable. The formulas are part of the
  architecture, encoded in the paper's equations 7, 9, and 10.

The ablation ("drop from `kept_path_scalars`") only closes Door 1. Door 2 is
open in every variant, including the "irrelevant" ones. That is the reason
the principled-vs-irrelevant contrast at k=30 shows only ≈1 pp of MAPE spread
and does not track the XAI ranking — not a bug of the pilot, not a
small-sample artefact, and not something a bigger dataset would fix.

### 1.5 What this justifies

- The retraining test is a valid measurement of **the learned embedding's
  contribution**, which is a real (and interesting) quantity.
- It is **structurally incapable** of measuring the importance of
  `traffic` / `packets` for the *whole* prediction, because those two features
  are load-bearing parts of the model's physics wrapper. That is a property of
  RouteNet-Fermi, not a deficiency of our protocol.

---

## 2. Complementary experiment: **perturbation** at inference time — RESULTS

The ablation cannot remove Door 2, but we can still stress it. If
`traffic` / `packets` really carry the signal the XAI methods say they do,
distorting their numeric values at inference must make MAPE explode.

### 2.1 Design (executed)

Model tested: the **frozen upstream RouteNet-Fermi baseline** (checkpoint
`traffic_models/delay/ckpt_dir_all_multiplexed/50-4.64`) — the exact model on
which IG and KernelSHAP produced the Step 4 attributions. No retraining. The
retrained pilot weights were unavailable (excluded from git by
`.gitignore`), which turned out to be a benefit: the fidelity test now runs
on the same model that XAI attributed over, so there is no train/test
distribution-shift confound.

Perturbation mode: **within-simulation flow shuffle** of one or more input
features. For each test simulation we generate a random permutation `π` of
the flow indices, then replace `x[f]` with `x[f][π]` for every `f` in the
selected feature set. Advantages:

- Marginal distribution is preserved exactly (each value is a real value
  from a real flow in the same simulation).
- When multiple features are shuffled with the **same** `π`, the per-flow
  ratio `traffic / packets = pkt_size` stays realistic (each flow gets
  another flow's realistic packet size).
- No distribution-shift complaint applies.

Executed by [`run_step7_perturbation.py`](../run_step7_perturbation.py) on the same 300-sim
test set used by Step 4, with 5 shuffle seeds per mode.

### 2.2 Results

Clean baseline = raw upstream model on the 300-sim test set, no perturbation.

| # | Mode | Features shuffled (IG rank) | MAPE (mean ± std) | ΔMAPE vs clean |
|---|---|---|---|---|
| 1 | `clean` | — | **31.85 %** | — |
| 2 | `shuffle_traffic` | traffic (#2) | 59.55 % ± 0.34 | **+27.71 pp** |
| 3 | `shuffle_packets` | packets (#3) | 51.10 % ± 0.15 | **+19.25 pp** |
| 4 | `shuffle_traffic_packets` | traffic + packets (same π) | 39.23 % ± 0.27 | +7.38 pp |
| 5 | `shuffle_top3` | sigma + traffic + packets (same π) | 38.54 % ± 0.28 | **+6.69 pp** |
| 6 | `shuffle_eq_lambda` | eq_lambda (#5) | 31.79 % ± 0.01 | −0.05 pp |
| 7 | `shuffle_avg_pkts_lambda` | avg_pkts_lambda (#10) | 31.73 % ± 0.01 | −0.11 pp |

Machine-readable outputs at
[`results/step7_perturbation_summary.csv`](step7_perturbation_summary.csv),
[`.json`](step7_perturbation_summary.json),
[`.md`](step7_perturbation_summary.md).

### 2.3 Interpretation

**Positive fidelity signal, on the same model XAI attributed over.**

- Every mode that touches the top-3 features (rows 2, 3, 4, 5) produces a
  substantial positive ΔMAPE — between **+6.7 and +27.7 percentage points**,
  a 20–90 % relative degradation of the model's error.
- The two bottom-7 controls (rows 6, 7) using the same procedure, same model,
  same distribution give **±0.1 pp** — indistinguishable from noise
  (seed-to-seed std = 0.01 pp).
- The contrast is **1–2 orders of magnitude**, mirroring the ≈19×
  attribution cliff seen in `rankings/ig.csv` between rank-3 and rank-4.

**Why the single-feature shuffles (rows 2, 3) hurt more than the joint one
(row 5) — a mechanistic finding, not a contradiction.**

- Row 5 uses the *same* permutation for all three top features, so flow *i*
  receives flow *j*'s complete scalar profile: its `sigma`, its `traffic`,
  its `packets`. The ratio `traffic/packets = pkt_size` is therefore flow
  *j*'s realistic packet size, and `sigma` is coherent with the other two.
  Only the mapping *(flow → routing graph)* is broken — the *values* remain
  internally consistent.
- Row 2 shuffles `traffic` alone. Flow *i* now has *its own* `packets` but
  flow *j*'s `traffic`, so `pkt_size = flow_j_traffic / flow_i_packets` —
  an impossible number that never appears in the dataset. The huge +27.7 pp
  reflects the model breaking on this per-flow physical inconsistency.
- Row 3 is the same story for `packets`.
- Row 4 (traffic + packets together, same π) preserves `pkt_size` but breaks
  `load`. Row 5 additionally shuffles `sigma` under the same π, which stays
  coherent → similar ΔMAPE.

This directly confirms the two-doors story quantitatively: the model is
sensitive to `traffic` / `packets` *values* precisely through the physics
wrapper (`load`, `pkt_size`, `trans_delay`).

**Bottom-7 controls behave exactly as XAI predicts.**

Shuffling `eq_lambda` or `avg_pkts_lambda` — features the ranking placed at
positions #5 and #10 with attribution scores >19× smaller than the top-3 —
does not move MAPE at all. If the XAI ranking were arbitrary, there would
be no reason for the bottom-7 to give such a clean null. This is the
positive-signal counterpart to the null-ablation
result — together, they form a defensible faithfulness story.

**Figure 3** below sketches the experimental pipeline.

---

## 3. Where is the machine learning in the delay prediction?

The final line of `delay_model.py` is misleading:

```python
return queue_delay + trans_delay          # line 194
```

That looks like the model is "just adding two numbers." It isn't. Here is
where each term comes from.

### 3.1 `trans_delay` — pure physics, zero learned parameters

```python
pkt_size    = traffic / packets                             # line 122
trans_delay = pkt_size * sum(1 / capacity_gather, axis=1)   # line 192
```

Average packet size ÷ link bandwidth, summed over hops. This is textbook
transmission time. **No neural network is involved.**

### 3.2 `queue_delay` — Little's Law applied to a *learned* queue occupancy

```python
queue_delay = sum(occupancy_gather / capacity_gather, axis=1)   # line 190
```

The `/ capacity` and the `sum` are physics (Little's Law: waiting time =
queue occupancy ÷ service rate, summed along the path). But
`occupancy_gather` is **the neural network's output**:

```python
occupancy_gather = self.readout_path(input_tensor)   # line 186
```

`readout_path` is a 3-layer MLP defined at lines 86–93. It predicts, for
each (flow, hop) pair, the number of bits of that flow queued at that hop.

### 3.3 What produces the input to `readout_path`

The final path hidden state, obtained after **T = 8 message-passing
iterations** (lines 151–181) of three coupled GRU cells:

- `path_update` — updates each path's state from the queues + links it
  traverses.
- `queue_update` — updates each queue's state from the paths crossing it.
- `link_update` — updates each link's state from its queues.

The **initial** path state comes from `path_embedding` (line 140), the MLP
that consumes the *learnable* per-flow scalars — **this is the layer that IG
and KernelSHAP attribute over**. Structural features (`length`, `model`,
`capacity`, `queue_size`, `policy`, `priority`, `weight`) initialise the
link/queue embeddings the same way. All of those are always present; none of
them are XAI candidates.

### 3.4 End-to-end picture

**Figure 2** below shows the full pipeline with the ML region and the
physics region colour-coded, and marks the exact tensor that XAI attributes
over.

### 3.5 How much does the ML actually contribute?

From `results/delay_decomposition.json`, computed on the 300-simulation test
set (81,600 flows):

| Quantity | Mean | Median | Notes |
|---|---|---|---|
| Predicted delay (`queue + trans`) | **0.957 TU** | 0.354 TU | model output |
| `trans_delay` (pure physics) | **0.131 TU** | 0.125 TU | no ML |
| `queue_delay` (ML → occupancy → /capacity) | **0.826 TU** | 0.217 TU | ML contribution |
| `trans_delay` share of prediction | **38.2 %** (mean) | 32.5 % (median) | up to 100 % on lightly-loaded flows |

Two things follow from this decomposition:

1. **The ML is not decorative** — on the mean flow, ~62 % of the predicted
   delay comes from the learned readout, not from the fixed physics.
2. **But on lightly-loaded flows the physics dominates** (up to 100 % of the
   prediction on the least-queued flows). This is the reason the retraining
   test's MAPE moves so little when we swap embedding features: a large
   fraction of the loss surface is decided by Door 2, which every variant
   shares.

This last point is the *quantitative* version of the two-doors story, and
turns the pilot's flat MAPE table from a puzzle into a predictable result.

---

## 4. Summary — three claims, three pieces of evidence

| Claim | Evidence |
|---|---|
| `traffic` / `packets` cannot be removed by the ablation | `_DROPPABLE_PATH_SCALARS` (data_generator.py:32); `load` / `pkt_size` / `trans_delay` reads (delay_model.py:120, 122, 192); arithmetic argument (§1.3) |
| Their importance is nonetheless demonstrable via a **perturbation** experiment on the frozen baseline | §2 — shuffling top-3 features on the same model IG attributed over blows MAPE from 31.85 % up to 59.55 % (+27.71 pp for `traffic`, +19.25 pp for `packets`); shuffling XAI-bottom-ranked features (eq_lambda, avg_pkts_lambda) gives ±0.1 pp (indistinguishable from noise). 1–2 orders of magnitude gap, matching the ≈19× attribution cliff. |
| The ML predicts **queue occupancy**, not delay directly | Trace `queue_delay ← occupancy_gather ← readout_path ← 8× message passing ← path_embedding`; 62 % of mean predicted delay is the ML contribution (`delay_decomposition.json`) |

---

## Figures (draw.io)

- **Figure 1 — Two doors of `traffic` / `packets`.** Shows the split
  between the learnable path embedding (drop-able) and the fixed physics
  computations (mandatory).
- **Figure 2 — Locus of learning.** End-to-end model with ML region and
  physics region colour-coded, marking the tensor XAI attributes over.
- **Figure 3 — Perturbation experiment.** Inference-time noise / shuffle /
  mean-replace pipeline that complements the ablation for
  `traffic` / `packets`.

Each figure is rendered as an interactive draw.io diagram alongside this
report and can be exported to `.drawio` / SVG / PNG.
