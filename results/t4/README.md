# results/t4 — Tesla T4 run (matches the shipped notebook)

This directory holds the aggregate evidence from the **Colab Tesla T4** execution of the final
experiment, captured in `finetuning_tutorial_executed_run2_t4.ipynb`.

**This run matches the currently distributed `finetuning_tutorial.ipynb`** — it loads
`unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` and resolves the revision from that same repository
(Python 3.12.13, torch 2.11, fp16). By contrast, `results/a5000/` was produced by an earlier
loader (`unsloth/Qwen2.5-1.5B-Instruct` + `use_exact_model_name=True`, Python 3.13, torch 2.10,
bf16). Both reach the same conclusions; see §11.5 of the tutorial for the side-by-side comparison.

## Contents
- `metrics.json` — all reported aggregate metrics (training, ROUGE, structured-JSON, challenge set).
- `environment.json` — GPU, precision, Python/torch, model repo + revision, key package versions.
- `loss_curves.png` — training and validation loss.

## Provenance / what is missing
These files were **reconstructed from the printed cell outputs** of the executed T4 notebook. The
per-example prediction logs (`*_predictions.jsonl`) were written inside the Colab runtime and were
not exported, so they are **not** reproduced here — the complete `results/a5000/` set has them, and
the T4 predictions remain embedded as cell outputs in `finetuning_tutorial_executed_run2_t4.ipynb`.
Aggregate numbers here are copied verbatim from the notebook's printed JSON, not re-estimated.
