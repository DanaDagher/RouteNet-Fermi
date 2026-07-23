# GPU Runbook — Step 7 (2 models + perturbation) on Sogeti RTX 4090

**You run every command. I do not connect to the server.**

**Target machine:** `sogeti147.tplinkdns.com:2222`, user `travel`, pwd `veltra147`.
Single-user GPU — **announce on Teams before you start** and again when you finish.

**Data quota:** 200 GB / month. Total transfer for this run is ≈ 4 GB up + a few MB down. Safe.

**Wall-clock estimate:** 2–3 hours end-to-end (30 min transfer, 40–80 min training, 15 min perturbation, 30 min copy-back + report).

**Design summary** (agreed with Dana, 2026-07-23):
- **2 retrained models on GPU** — baseline (all 10 features) and principled_relevant (top-3 only).
- **1 perturbation pass** on the retrained baseline — shuffles top-3 features {sigma, traffic, packets} to run the "necessity" test that ablation cannot (traffic and packets can't be dropped from the model because the physics formulas need them).
- The retrained principled_irrelevant model is **not needed**: shuffling the top-3 on the baseline at inference achieves the same goal without the two-doors caveat.
- Random controls are **not needed**: perturbation on XAI-bottom-ranked features (eq_lambda, avg_pkts_lambda) gives fidelity controls.

---

## Phase A — Local prep on your Windows PC (~10 min)

Do these from `cmd.exe` at the repo root (`C:\Users\ddagher\RouteNet-Fermi`), on branch `xai-protocol-b`.

### A.1 Commit and push the new files to your fork

Your fork is `https://github.com/DanaDagher/RouteNet-Fermi.git` (safe to push). **Do NOT push to BNN-UPC upstream** — you only push to your own fork.

```cmd
git status
git add run_step7_gpu_2models.sh run_step7_perturbation.py results\step7_perturbation_summary.csv results\step7_perturbation_summary.json results\step7_perturbation_summary.md results\two_doors_and_ml_locus_REPORT.md GPU_RUNBOOK.md
git commit -m "Add GPU 2-model Step 7 script, perturbation experiment, two-doors report"
git push origin xai-protocol-b
```

Verify on GitHub that the push landed on **`DanaDagher/RouteNet-Fermi` branch `xai-protocol-b`**.

### A.2 (Optional) Quick sanity check that the 2-model script parses

```cmd
bash -n run_step7_gpu_2models.sh
```

No output = OK. (If `bash` is not on your PATH, skip — the syntax is trivial and will run fine on Linux.)

---

## Phase B — Connect to the GPU server (~5 min)

You have two options. Pick one and stick with it.

### Option B1 — Built-in Windows OpenSSH (recommended for cmd users)

Windows 10/11 ships with an SSH client. From `cmd.exe`:

```cmd
ssh -p 2222 travel@sogeti147.tplinkdns.com
```

Enter password `veltra147`. First time it asks about a host key — type `yes`.

To avoid re-typing the password every time (optional but nice), generate a key once and copy it:

```cmd
ssh-keygen -t ed25519 -C "dana-routenet"
type %USERPROFILE%\.ssh\id_ed25519.pub | ssh -p 2222 travel@sogeti147.tplinkdns.com "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

### Option B2 — PuTTY (what the Sogeti email suggested)

Open PuTTY → Session → Host `sogeti147.tplinkdns.com`, Port `2222`, SSH → Open. User `travel`, pwd `veltra147`.

For file transfer with PuTTY you use `pscp.exe` (comes with PuTTY). If you install PuTTY, add it to PATH — `pscp` calls below assume it is on PATH; otherwise substitute the full path.

---

## Phase C — First-time setup on the server (~10 min, only once)

**Once you are inside the SSH session** (`travel@sogeti147:~$` prompt), run the following on the SERVER.

### C.1 Check the GPU is there and healthy

```bash
nvidia-smi
```
You should see one RTX 4090 with 24 GB memory. If the GPU is busy (someone else's process), **exit and message Teams** — do not proceed.

### C.2 Check Python & make a personal workspace

```bash
which python3 && python3 --version
mkdir -p ~/dana && cd ~/dana
```

The project requires **Python 3.7–3.9** (TensorFlow 2.6.x pinned in `requirements.txt`). If `python3 --version` shows 3.10+, prefer `python3.9` or `python3.8` instead — if none exist, install with `pyenv` (ask Arthur, don't sudo).

### C.3 Clone your fork

```bash
cd ~/dana
git clone -b xai-protocol-b https://github.com/DanaDagher/RouteNet-Fermi.git
cd RouteNet-Fermi
git log --oneline -3
```
The top commit should be the one you pushed in Phase A.

### C.4 Create the virtualenv and install requirements

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Verify TensorFlow sees the GPU:**
```bash
python -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```
Expected: `TF: 2.6.x` and `GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`.

If GPUs list is empty, CUDA/cuDNN aren't wired up for TF 2.6. Either install `tensorflow-gpu==2.6.*` explicitly, or ask Arthur for the CUDA version already on the box. **Do not proceed without a visible GPU** — otherwise the run will silently fall back to CPU and take days.

---

## Phase D — Transfer the dataset from your PC to the server (~15–30 min)

The `data/` folder is git-ignored (~13 GB total). You only need the `all_multiplexed` sub-dataset for this run (~4 GB).

From **your local Windows cmd.exe** (NOT inside the ssh session), open a *second* terminal and run:

```cmd
cd C:\Users\ddagher\RouteNet-Fermi
scp -P 2222 -r data\traffic_models\all_multiplexed travel@sogeti147.tplinkdns.com:/home/travel/dana/RouteNet-Fermi/data/traffic_models/
```

That copies `train/`, `test/`, `validation/` under one folder. Enter the password when prompted.

**If scp isn't on your PATH** (older Windows), use `pscp` from PuTTY:
```cmd
pscp -P 2222 -r data\traffic_models\all_multiplexed travel@sogeti147.tplinkdns.com:/home/travel/dana/RouteNet-Fermi/data/traffic_models/
```

### D.1 Verify on the server

Back in the SSH session:
```bash
du -sh ~/dana/RouteNet-Fermi/data/traffic_models/all_multiplexed/{train,test,validation}
```
Expected: ≈ 2.2 GB / 962 MB / (validation size). If any dir is missing, re-scp just that one.

---

## Phase E — Launch the 2-model training (~40–80 min)

### E.1 **Post on Teams**: *"Starting GPU run on sogeti147, ~1h30."*

### E.2 Start training inside a screen or tmux so an SSH drop doesn't kill it

```bash
cd ~/dana/RouteNet-Fermi
source venv/bin/activate
screen -S step7
chmod +x run_step7_gpu_2models.sh
./run_step7_gpu_2models.sh 2>&1 | tee results/_step7_gpu.log
```

**Detach from screen** without stopping it: press `Ctrl+A` then `d`. You are back at the shell prompt but training keeps going.

**To re-attach later** (even from a fresh SSH session):
```bash
screen -r step7
```

### E.3 Monitor progress in a second SSH window

Open a second `cmd.exe`, ssh in again, then:
```bash
tail -f ~/dana/RouteNet-Fermi/results/_step7_gpu.log
```
Each cell prints one line per epoch. You should see loss going down. `nvidia-smi -l 5` in a third window shows live GPU utilisation (expect 60–90 % during training).

### E.4 Sanity gate — after cell 1 (baseline) finishes

Roughly 20–40 min in, cell 1 writes `checkpoints/gpu_full/baseline_seed42/metrics.json`. Peek:
```bash
cat ~/dana/RouteNet-Fermi/checkpoints/gpu_full/baseline_seed42/metrics.json
```
`test_mape` should be in a reasonable range (**5–15 %** for full-data 150-epoch baseline; anything above 30 % means normalization or data path is wrong — stop and diagnose before cell 2 wastes GPU time).

---

## Phase F — Perturbation pass on the retrained baseline (~15 min)

**Only after both cells are done.** This is the **necessity test** — it shuffles the top-3 features (sigma, traffic, packets) at inference on the retrained baseline, achieving the "remove the important features" experiment that ablation cannot.

From inside the SSH session:
```bash
cd ~/dana/RouteNet-Fermi
source venv/bin/activate
```

**No edits required** — `run_step7_perturbation.py` already targets `checkpoints/gpu_full/baseline_seed42` and runs the full mode list. Just launch:
```bash
python run_step7_perturbation.py --pool 300 --n-seeds 5 2>&1 | tee results/_step7_perturbation_gpu.log
```

### F.1 Rename any pre-existing perturbation table you want to keep

The script writes to `results/step7_perturbation_summary.{csv,json,md}` and will overwrite the frozen-upstream-baseline table produced locally. If you want both:
```bash
mv results/step7_perturbation_summary.csv  results/step7_perturbation_summary_upstream.csv
mv results/step7_perturbation_summary.json results/step7_perturbation_summary_upstream.json
mv results/step7_perturbation_summary.md   results/step7_perturbation_summary_upstream.md
```
Then run the perturbation script (the new outputs will be named `step7_perturbation_summary.*` and will be the GPU baseline set).

### F.2 Modes that run on the GPU baseline (all 10 features present)

| Mode | Features shuffled | Purpose |
|---|---|---|
| `clean` | — | reference MAPE |
| `shuffle_sigma` | sigma (#1) | top-3 individual |
| `shuffle_traffic` | traffic (#2) | top-3 individual |
| `shuffle_packets` | packets (#3) | top-3 individual |
| `shuffle_traffic_packets` | traffic + packets (same π) | keeps pkt_size realistic |
| `shuffle_top3` | sigma + traffic + packets (same π) | **full necessity test** |
| `shuffle_eq_lambda` | eq_lambda (#5) | bottom-7 control |
| `shuffle_avg_pkts_lambda` | avg_pkts_lambda (#10) | bottom-7 control |

All 8 modes run on the baseline. If XAI is faithful: the 5 top-3 modes give large positive ΔMAPE, the 2 bottom-7 controls give ~0. That IS the fidelity result.

---

## Phase G — Save the weights and copy results back (~15 min)

### G.1 On the server: tar the weights (they are git-ignored)

```bash
cd ~/dana/RouteNet-Fermi
tar czf gpu_full_checkpoints.tgz checkpoints/gpu_full/
du -sh gpu_full_checkpoints.tgz
```

Expect ~100–300 MB. Keeping the tar means Steps 8+ don't hit the "weights lost" problem.

### G.2 From your local cmd.exe, pull results back

```cmd
cd C:\Users\ddagher\RouteNet-Fermi
scp -P 2222 travel@sogeti147.tplinkdns.com:/home/travel/dana/RouteNet-Fermi/gpu_full_checkpoints.tgz .
scp -P 2222 -r travel@sogeti147.tplinkdns.com:/home/travel/dana/RouteNet-Fermi/checkpoints/gpu_full checkpoints\
scp -P 2222 -r travel@sogeti147.tplinkdns.com:/home/travel/dana/RouteNet-Fermi/results\* results\
```

The `.tgz` is your safety archive — stash it on OneDrive/USB after the run. The `checkpoints/gpu_full/` copy is what future scripts will read.

### G.3 Commit the metrics + results (NOT the weights)

The updated `.gitignore` already excludes `*.index`, `*.data-*`, and `checkpoint`, so `git add checkpoints/gpu_full/` will only pick up the small JSON/CSV files. Verify with `git status` before committing:

```cmd
git add checkpoints\gpu_full results\_step7_gpu.log results\_step7_perturbation_gpu.log results\step7_perturbation_summary*
git status
git commit -m "GPU 2-model rerun: baseline + principled_relevant + top-3 perturbation on baseline"
git push origin xai-protocol-b
```

If `git status` shows any `.data-*` or `.index` files, run `git reset HEAD <file>` — DO NOT commit binary weights to git.

---

## Phase H — Wrap up (~10 min)

1. **Post on Teams**: *"GPU freed, run finished."*
2. On the server, exit screen and log out:
   ```bash
   exit          # exits the screen if attached
   exit          # exits the SSH session
   ```
3. Update `results/two_doors_and_ml_locus_REPORT.md` locally with the new GPU numbers (the perturbation table gets a "GPU retrained baseline" section; the frozen-upstream table becomes a cross-check).
4. Stash `gpu_full_checkpoints.tgz` on OneDrive.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `nvidia-smi` says no devices | Someone else killed the driver or a reboot pending | Ask on Teams; do not proceed. |
| `tf.config.list_physical_devices('GPU')` returns `[]` | CUDA/cuDNN missing for TF 2.6 | `pip install nvidia-cudnn-cu11` or ask Arthur for the pre-installed CUDA version. |
| Training MAPE > 30 % after cell 1 | Data path wrong, or a normalization/z-score mismatch | Check `du -sh data/traffic_models/all_multiplexed/train` matches your local (~2.2 GB); look at `training_log.csv` for the loss curve. |
| SSH drops mid-run | Wi-Fi / network | Screen keeps training alive. Re-ssh, then `screen -r step7`. |
| Out of disk on the server | Weights + cache + dataset ≈ 6–8 GB | Delete `checkpoints/gpu_full/**/_train_cache*` after run (see `.gitignore` line 12) — those are tf.data caches, safe to remove. |
| Script skips a cell you want to rerun | `metrics.json` exists in that cell's output dir | `rm -rf checkpoints/gpu_full/<that_cell>` then relaunch — resume-safe. |

---

## Etiquette recap (Sogeti rules)

- **One user on GPU at a time.** Announce on Teams before you start and when you finish.
- **200 GB / month data quota.** This run uses ≈ 5 GB total (4 GB up, 1 GB down max). Safe.
- **Your workspace only.** Everything under `~/dana/` on the server. Don't touch other users' dirs.
- **No `sudo`** unless Arthur said so.
- **Only push to your fork** (`DanaDagher/RouteNet-Fermi`). Never push to `BNN-UPC/RouteNet-Fermi` upstream — see memory rule.
