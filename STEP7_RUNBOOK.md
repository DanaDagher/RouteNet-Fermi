# Step 7 Runbook — Retraining the 13-Cell Matrix on the Sogeti RTX 4090

> ⚠️ **SUPERSEDED (2026-06-17) — retained for reference only.**
> Compute moved from the Sogeti RTX 4090 to **GCP** (company-provided). The
> conda/CUDA setup below is Sogeti-specific; a **GCP runbook will be written
> once the VM image / GPU type are confirmed** (not yet decided).
> Also note the matrix changed: the 13 cells are now **1 baseline + 6 principled
> (IG ≡ KernelSHAP, from `configs/ig/`) + 6 random** (output dirs
> `checkpoints/principled/` and `checkpoints/random/`), per the updated
> `run_step7_all.sh`. The hyperparameters and the "Dana runs it by hand, Claude
> never touches the remote machine" rule are unchanged. See THESIS_DECISIONS §7.

**Who runs this:** Dana, by hand, on the GCP VM (formerly the Sogeti machine).
Claude Code never accesses the remote machine — it only prepares scripts and
analyzes what you bring back.

**What it produces:** 13 trained models under `checkpoints/`, each with
`metrics.json`, `training_log.csv`, `training_config.json`, and per-epoch
weight checkpoints. These feed Step 8 (fidelity analysis).

**Locked hyperparameters (every cell):** 150 epochs × 2,000 samples,
Adam lr=0.001, MAPE loss, hidden 32, T=8, seed 42.

---

## Phase 0 — One-time machine setup

### 0.1 Install Miniconda (skip if `conda` already exists)

```bash
which conda || {
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
  bash ~/miniconda.sh -b -p ~/miniconda3
  ~/miniconda3/bin/conda init bash
  exec bash   # reload shell
}
```

### 0.2 Create the environment (TF 2.6.5 — same as the Windows laptop)

The RTX 4090 driver speaks CUDA 12.8, but TF 2.6.5 needs CUDA 11.2 libraries.
Conda ships them inside the env; the driver is backward-compatible (the first
run JIT-compiles kernels for the 4090, so expect a few extra minutes once).

```bash
conda create -n rnf python=3.9 -y
conda activate rnf
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1 -y
pip install "tensorflow==2.6.5" "keras==2.6.0" "protobuf<3.20" "networkx==2.6.3" "pandas==1.1.5"

# make the env's CUDA libs visible on activation
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:${LD_LIBRARY_PATH:-}' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
conda deactivate && conda activate rnf
```

### 0.3 Verify the GPU is visible to TF

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

**Expected:** `2.6.5` and one `PhysicalDevice ... GPU:0`. If the GPU list is
empty, copy the full output (it names the missing library) and send it to me.

### 0.4 Clone the repo (branch `xai-protocol-b`)

```bash
cd ~
git clone --branch xai-protocol-b https://github.com/DanaDagher/RouteNet-Fermi.git
cd RouteNet-Fermi
git log --oneline -2   # should show the Step 6/7 commits
chmod +x run_step7_all.sh
```

### 0.5 Transfer the dataset (3.2 GB)

From **your Windows laptop** (PowerShell, from `C:\Users\ddagher\RouteNet-Fermi`):

```powershell
scp -r data\traffic_models\all_multiplexed travel@<SOGETI_IP>:~/RouteNet-Fermi/data/traffic_models/
```

(Replace `<SOGETI_IP>` with the machine's address. Alternative: download
`dataset-v6-traffic-models` from the BNN-UPC site directly on the machine.)

Then verify on the Sogeti machine:

```bash
ls ~/RouteNet-Fermi/data/traffic_models/all_multiplexed/train ~/RouteNet-Fermi/data/traffic_models/all_multiplexed/test
du -sh ~/RouteNet-Fermi/data/traffic_models/all_multiplexed
```

**Expected:** `train` shows `geant2-multiplexed nsfnet-multiplexed`, `test`
shows `gbn-multiplexed`, total ~3.2G.

---

## Phase 1 — Pre-flight checks (10–20 min)

### 1.1 Smoke test (same one that passed 13/13 on Windows)

```bash
cd ~/RouteNet-Fermi
python run_step6_smoke.py
```

**Expected:** ends with `Total: 13/13 PASS`. Anything else → stop, send me
the output.

### 1.2 Dry-run training (2 epochs × 50 steps, throwaway output)

```bash
export PYTHONHASHSEED=42
python run_step7_train.py --config configs/baseline/full.json \
    --output /tmp/step7_dry --epochs 2 --steps-per-epoch 50 --validation-steps 20 --test-steps 20
cat /tmp/step7_dry/metrics.json
rm -rf /tmp/step7_dry
```

**Expected:** runs to completion, prints a test MAPE (value meaningless after
2 epochs), writes metrics.json. While it runs, check the GPU is actually
working in a second terminal: `nvidia-smi` should show a python process.

---

## Phase 2 — Baseline cell first (sanity gate)

Train the baseline alone before committing to the full matrix:

```bash
cd ~/RouteNet-Fermi
export PYTHONHASHSEED=42
nohup python run_step7_train.py --config configs/baseline/full.json \
    --output checkpoints/baseline_seed42 > baseline.log 2>&1 &
tail -f baseline.log        # Ctrl-C detaches from tail only, training continues
```

When it finishes:

```bash
cat checkpoints/baseline_seed42/metrics.json
```

**GATE:** `test_mape` should land in the vicinity of the paper's 4.71%
(Step 2.5 reproduced 4.62% with the upstream pretrained checkpoint). A freshly
trained seed-42 model won't match exactly — **anything in ~3.5–6% is fine**.
If it's wildly off (>2 pp), stop and send me `metrics.json` +
`training_log.csv` before burning GPU-days on the other 12 cells.

**Also note `training_time_minutes`** — multiply by 13 for the full-matrix
estimate, and tell me the number so we can check it against the deadline.

---

## Phase 3 — The full matrix

The driver skips the already-done baseline automatically (it checks for
`metrics.json` per cell), so just launch:

```bash
cd ~/RouteNet-Fermi
nohup ./run_step7_all.sh > step7_all.log 2>&1 &
```

Monitoring (any time):

```bash
tail -20 step7_all.log                          # overall progress
ls checkpoints/*/metrics.json checkpoints/*/*/metrics.json 2>/dev/null | wc -l   # cells done (target: 13)
grep -E "^# CELL|CELL OK|CELL FAILED" step7_all.log
nvidia-smi                                       # GPU alive?
```

**Interruptions are safe.** If the machine reboots or the run dies, just
re-run `nohup ./run_step7_all.sh > step7_all2.log 2>&1 &` — completed cells
are skipped, and a half-trained cell resumes from its latest epoch checkpoint.

---

## Phase 4 — Bring the results back

When `step7_all.log` ends with `All cells completed.`:

### 4.1 Small files (metrics + logs) — these get committed to git

On the Sogeti machine:

```bash
cd ~/RouteNet-Fermi
tar czf step7_results.tar.gz step7_all.log baseline.log \
    $(find checkpoints -name 'metrics.json' -o -name 'training_log.csv' -o -name 'training_config.json')
```

On your Windows laptop:

```powershell
scp travel@<SOGETI_IP>:~/RouteNet-Fermi/step7_results.tar.gz C:\Users\ddagher\RouteNet-Fermi\
```

Then tell me — I'll extract, verify all 13 cells, build the Step 7 report,
and commit. **Do not delete anything on the Sogeti machine yet.**

### 4.2 Weight checkpoints (large) — keep, don't commit

The per-epoch checkpoints stay on the Sogeti machine as the archive of record
(Step 8 only needs the metrics, but keep the weights until the thesis is
done in case we need to re-evaluate a model).

---

## Phase 5 — Random control (CONDITIONAL — do not start without discussing)

Only if all 13 cells completed cleanly AND there is GPU time to spare
(PIPELINE §7.B). The 6 random-control configs don't exist yet — I generate
them from `rankings/random.csv` when we decide to go. Decision point: after
Phase 4 review.

---

## If something goes wrong

| Symptom | What to do |
|---|---|
| GPU list empty in 0.3 | Send me the import output; likely LD_LIBRARY_PATH or cudnn version |
| `CUDA_ERROR_*` / cuDNN errors during dry-run | Send the traceback — fallback plan is TF 2.10.1 in the same env recipe (I'll give exact commands) |
| First epoch extremely slow | Normal once (PTX JIT for the 4090); worry only if epoch 2+ is also slow |
| A cell shows `CELL FAILED` | Keep the matrix running; send me the tail of step7_all.log around the failure |
| Baseline MAPE way off the gate | Stop after baseline; send metrics.json + training_log.csv |
| Disk fills up | `du -sh checkpoints/` — 150 epochs × 13 cells of weights is a few GB; tell me before deleting anything |
