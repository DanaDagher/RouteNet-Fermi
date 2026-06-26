# results/

Write-ups, metrics, and figures produced at each pipeline step.

| File/dir | What it is |
|---|---|
| `step1_environment_report.md` | Step 1: environment/setup notes. |
| `step2_5_validation_report.md` | Step 2.5: reproduction of the upstream paper's Table V on all 5 sub-datasets, using the frozen pretrained checkpoints (validity check only). |
| `baseline_validation/` | Per-sub-dataset MAPE results backing the Step 2.5 report. |
| `step3_report.md` | Step 3: XAI method implementation + sanity checks. |
| `step4_report.md` | Step 4: IG + KernelSHAP explanations generated on N=300 `all_multiplexed` test simulations. |
| `inference/` | Raw/intermediate outputs from Step 4 (per-method explanation runs, stability runs, audit logs). |
| `step5_report.md` | Step 5: global feature ranking aggregation + half-split Spearman stability check. |
| `step6_report.md` | Step 6: column-dropping variant config generation + smoke test. |
| `step7_scope_decision.md` | Decision record narrowing Step 7's scope (compute constraints). |
| `step7_pilot_subdataset.md`, `step7_pilot_report_DRAFT.md` | Step 7 pilot: CPU run on a reduced (N=500) subset, 5 of the 13 models. **This is the final Step 7 result** — no GPU rerun, no k=50/k=70 sweep was executed; see the report for why this is itself the key finding (the "two-door" feature-entanglement issue makes the faithfulness test inconclusive by design). |
| `step7_delay_mechanism_analysis_DRAFT.md` | Supporting analysis of RouteNet-Fermi's queuing-theory structure explaining the Step 7 result. |
| `theoretical_expectations.md` | Pre-registered expectations written before seeing Step 7 results. |
| `design_decisions_log.md` | Running log of methodological decisions made during the pipeline (supplements `THESIS_DECISIONS.md`). |
| `training_stats.json` | Aggregated training statistics across retrained models. |
| `deliverable_subsections_DRAFT.md`, `deliverable_subsections_Draftv2.md` | Drafts of the prose later assembled into the D3.2 deliverable (see `deliverable/`). |
| `figures/` | Final figures used in the deliverable (attribution cliff, faithfulness MAPE comparison, IG-vs-KernelSHAP radar chart). |
