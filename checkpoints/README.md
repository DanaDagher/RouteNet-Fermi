# checkpoints/

Retrained model checkpoints, one per variant defined in `configs/`. Large TF weight files
(`*.data-*`, `*.index`, `checkpoint`, `_train_cache*`, `_val_cache*`) are gitignored — only
small metric/log files are tracked here. To regenerate the weights, rerun the corresponding
`run_step7_*` script against the matching config.

| Dir | What it is |
|---|---|
| `pilot_n500/` | The CPU pilot run: N=500 training subset (not the full `all_multiplexed` training split), 5 of the 13 planned models — `baseline_seed42/`, `principled/k30_relevant/`, `principled/k30_irrelevant/`, `random/k30_relevant/`, `random/k30_irrelevant/`. This is the final, reported Step 7 result; see `results/step7_pilot_report_DRAFT.md` for scope and why it wasn't extended to the full dataset or to k=50/70. |
