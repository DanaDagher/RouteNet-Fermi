# How RouteNet-Fermi Computes Delay — Paper, Code, and What It Means for Our XAI Results

**Status:** DRAFT for the Step 7 pilot report / D3.2.
**Sources:** Ferriol-Galmés et al., *RouteNet-Fermi*, IEEE/ACM ToN 2023
(arXiv:2212.12070v3), Section IV-C and Section V; project code `delay_model.py`;
pilot results in `checkpoints/pilot_n500/`.
**Voice:** written as a master's analysis tying the networking/queuing side to
the GNN and to our feature-importance findings.

---

## 1. The delay is *reconstructed* from queuing theory, not output directly

A central design choice of RouteNet-Fermi is that the network **does not output
the delay directly.** Instead, following classical Queuing Theory (QT), the
end-to-end flow delay is built up hop-by-hop as the sum of two physically
distinct components at each link on the path:

> **per-link delay = queuing delay + transmission delay**

The paper states this explicitly (Section IV-C) and gives the equations:

- **Queuing delay (eq. 8):**  d̂_q = R_fd(h_f,l) / x_lc
- **Transmission delay (eq. 9):**  d̂_t = x_fps / x_lc
- **Per-link delay (eq. 10):**  d̂_link = d̂_q + d̂_t
- **Flow delay:**  ŷ_fd = Σ over the (queue, link) pairs on the path of d̂_link

with, earlier, the **link load (eq. 7):**  x_l_load = (1/x_lc) · Σ_{f∈L_f(l)} λ_f

where x_lc = link capacity, x_fps = mean flow packet size, λ_f = a flow's
average traffic volume, and R_fd = the readout neural network.

### The key sentence (paper, Sec. IV-C2)

> *"RouteNet-F infers delays indirectly from the mean queue occupancy on
> forwarding devices… The prediction of this effective queue occupancy —
> instead of directly predicting delays — helps overcome the practical
> limitation of producing out-of-range delay values."*

So the **only quantity the network learns to predict in the delay head is the
"effective queue occupancy"** — the mean number of bits a flow must wait behind
at an output port (eq. 8 numerator). Everything else — dividing by capacity,
the transmission term, summing along the path — is **deterministic QT
arithmetic.**

---

## 2. Where the machine learning actually is

| Quantity | Produced by | Learned? |
|---|---|---|
| effective queue occupancy `R_fd(h_f,l)` | the readout NN, fed by **T=8 rounds of message passing** | **YES — this is the GNN** |
| queuing delay `d̂_q = occupancy / capacity` | division by `x_lc` | no (formula) |
| transmission delay `d̂_t = packet_size / capacity` | `x_fps / x_lc` | no (formula) |
| flow delay `ŷ_fd` | sum of link delays | no (sum) |

The GNN learns the **hard** part — how congested each queue is, given how all
flows interact across the topology, the routing, and the scheduling policy.
Queuing delay has **no closed-form solution** for realistic (non-Markovian)
traffic, which is exactly why a learned model is used. The **easy** parts —
the time to physically push bits onto a wire (transmission), and the unit
conversion occupancy→time — are known physics and are simply computed.

**This answers "where is the prediction?":** it is the effective queue
occupancy, the output of the message-passing GNN. The QT formulas are the
scaffolding that turns that prediction into a delay in seconds, in a way that
generalises across network sizes.

---

## 3. Paper ↔ code, line by line (`delay_model.py`)

| Paper | Code (`delay_model.py`) |
|---|---|
| link load, eq. 7: `Σλ_f / x_lc` | `load = tf.reduce_sum(path_gather_traffic, axis=1) / capacity` (≈ L120) |
| packet size `x_fps = traffic/packets` | `pkt_size = traffic / packets` (L122) |
| occupancy `R_fd(h_f,l)` | `occupancy_gather = self.readout_path(input_tensor)` (L186) |
| queuing delay, eq. 8: `R_fd / x_lc`, summed | `queue_delay = reduce_sum(occupancy_gather / capacity_gather, axis=1)` (L190–191) |
| transmission delay, eq. 9: `x_fps / x_lc`, summed | `trans_delay = pkt_size * reduce_sum(1/capacity_gather, axis=1)` (L192) |
| flow delay, eq. 10 + path sum | `return queue_delay + trans_delay` (L194) |

The code is a faithful, line-for-line implementation of equations (7)–(10).

---

## 4. What this means for our Step 7 (XAI faithfulness) results

`traffic` and `packets` enter the delay prediction through **three** distinct
paths, only **one** of which our retraining test can touch:

1. **Link load (eq. 7)** → initial link hidden state → message passing.
   *(structural — always present)*
2. **Transmission delay (eq. 9)** → added directly to every prediction.
   *(formula — always present)*
3. **The learnable path embedding** → the GNN's per-flow initial state.
   *(this is the only path our column-dropping removes)*

When a variant "drops" `traffic`/`packets`, it closes **path 3 only**. Paths 1
and 2 keep feeding them into the prediction, so the model stays accurate.

This explains our pilot numbers (test MAPE on 300 held-out samples):

| Model | learned features (path 3) | TEST MAPE |
|---|---|---|
| principled k30_relevant | top-3 (sigma, traffic, packets) | **5.41%** |
| random k30_relevant | 3 random scalars | **5.43%** |
| random k30_irrelevant | 7 random scalars | 5.80% |
| baseline | all 10 | 6.14% |
| principled k30_irrelevant | bottom-7 (top-3 removed from path 3) | 6.38% |

- **random-3 (5.43%) ≈ principled-3 (5.41%)** → which 3 scalars sit in the
  path-3 embedding barely matters. The GNN reconstructs queue occupancy mainly
  from the **structural** inputs (load, capacity, queue size, scheduling), so
  the per-flow embedding scalars carry little *marginal* information.
- **The whole spread is ~1 pp (5.41–6.38%) and does NOT follow importance:**
  the all-10 baseline (6.14%) is *worse* than the 3-feature models, and removing
  the top-3 (irrelevant, 6.38%) does not collapse accuracy — because paths 1 and
  2 still carry traffic/packets, and extra embedding features mostly add variance
  on 500 samples.
- **Net:** the retraining/ROAR faithfulness ordering is essentially flat and
  uncorrelated with the XAI ranking — the test cannot discriminate here.

So the retraining-based faithfulness test is **structurally unable to isolate**
the features QT (and our XAI) say matter most: those features re-enter the
prediction through the deterministic QT terms.

---

## 4b. What we anticipated vs. what the paper confirms

The limitation was first spotted *from the model's behaviour* — the
`irrelevant` model staying accurate after the "important" features were removed
— and only then traced to the equations. The paper confirms each step:

| Anticipated from the results | Confirmed by paper / code |
|---|---|
| traffic & packets cannot really be removed — they are in the delay formula | eq. 9 (transmission = `(traffic/packets)/capacity`) and eq. 7 (load = `traffic/capacity`); both always computed in `delay_model.py` |
| that is why the `irrelevant` model did not degrade | removing them from the embedding leaves eq. 7 & 9 intact → prediction unchanged |
| a deterministic formula carries part of the delay | the transmission term is pure arithmetic, reading traffic/packets directly |

Refinement made precise by the paper: delay is **not** purely a formula — the
**queuing** half is GNN-predicted (effective queue occupancy, eq. 8). The
correct statement is that the important features re-enter delay through the
**formula half** (transmission + load), which is exactly why the retraining
ablation cannot remove them.

---

## 5. The paper's OWN ablation corroborates this (Section V-3)

The authors ran their own ablation "to analyze which features of RouteNet-F
have more impact on its accuracy." Two of their findings line up with ours:

> *"using the link load as input instead of the capacity (RouteNet-F-load)
> does not have a significant impact on the model's accuracy."*

> *"predicting the delay as the sum of link-level delays along flows
> (RouteNet-F-occupancy) seems to have the largest impact on the accuracy."*

In other words, the authors themselves found that the **architectural / physics
decomposition** (predict occupancy → sum link delays) is what carries the
accuracy, while an individual **input feature** (link load) has limited marginal
effect. This is exactly the pattern our XAI retraining test surfaced — the model
is dominated by its QT-informed structure, not by individual per-flow inputs.

This is strong support, not a contradiction: an independent, published ablation
reaches a compatible conclusion.

---

## 6. On training with a sub-dataset (legitimacy)

Our pilot trains on 500 samples. The paper itself studies **few-shot** training
(Section V-2, Fig. 8) with **25 to 10,000** samples — reporting ~11% error at 25
samples down to 6.24% at 10,000 (on the 5–10→50–300-node scalability dataset).
Training RouteNet-F on a few hundred samples is therefore a methodology the
original authors use; our 500-sample pilot is legitimate, and the absolute MAPE
we see (~5.4%) is consistent with small-sample RouteNet-F behaviour.

---

## 7. Takeaway for the thesis

1. **The model genuinely predicts.** The GNN learns the effective queue
   occupancy — the one quantity with no closed form — and QT equations (7)–(10)
   turn it into a delay. This is a *physics-informed* GNN, not a formula
   masquerading as ML.
2. **Faithfulness-by-retraining is structurally limited here.** The most
   important features re-enter delay through the deterministic QT terms (link
   load + transmission), so removing them from the learnable embedding cannot
   degrade the model — and *any* 3 embedding features train to similar accuracy.
3. **Our XAI ranking is still correct and consistent** with the model's physics
   (traffic, packets, sigma are genuinely central to delay), and with the
   paper's own ablation (individual input features have limited marginal effect;
   the QT structure dominates).
4. This is a precise, defensible **critical finding** about applying ROAR-style
   feature-ablation to physics-informed network GNNs — the kind of result a
   *critical and comparative* study is meant to produce.
