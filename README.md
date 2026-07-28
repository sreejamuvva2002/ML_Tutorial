# Fine-Tuning Large Language Models: LoRA and QLoRA

A hands-on tutorial that fine-tunes **Qwen2.5-1.5B-Instruct** on a slice of the public
Alpaca-cleaned instruction dataset using **QLoRA** — a frozen 4-bit base model with small
trainable low-rank adapters on top — and evaluates the result against its own un-tuned
baseline. Everything runs end to end on a **single free Google Colab T4 GPU (16 GB)**. No paid
compute, no API keys, no private data.

Course tutorial submission — ARTI 4555/6555, MS in Artificial Intelligence.
Author: Sreeja Muvva.

## Contents

| File | Purpose |
|---|---|
| [`Fine_Tuning_LLMs_Tutorial.md`](Fine_Tuning_LLMs_Tutorial.md) | The written tutorial: concepts, code, interpretation, exercises |
| [`finetuning_tutorial.ipynb`](finetuning_tutorial.ipynb) | The same code as an executable Colab/Jupyter notebook |
| [`finetuning_tutorial_executed.ipynb`](finetuning_tutorial_executed.ipynb) | A complete run on a T4 with all outputs and the loss curve |
| `requirements.txt` | Pinned dependencies for local (non-Colab) runs |

## Results

One full run on a free Colab T4 (2026-07-27, `unsloth 2026.7.5 / trl 0.24.0 / transformers 5.5.0`),
written up in §11.5 of the tutorial:

| | |
|---|---|
| LoRA trainable parameters | 18,464,768 (1.18%) — matches the §6.3 hand calculation exactly |
| Peak VRAM | 4.06 GB of 15.6 GB |
| Training time | 9.4 min (1,700 examples, 1 epoch, 213 steps) |
| Best validation loss / perplexity | 1.0211 / 2.78 |
| ROUGE-1 / 2 / L on held-out data | 0.525 / 0.306 / 0.401 |

The interesting finding is negative: **fine-tuning did not improve the model and slightly degraded
output formatting.** Qwen2.5-1.5B-Instruct is already instruction-tuned, and Alpaca is an older
dataset distilled from a weaker teacher — so SFT taught a strong model to imitate a worse one. The
loss fell while human-visible quality did not improve, which is the tutorial's own §10.3 warning
demonstrated on real output. See §11.5 for the side-by-side evidence.

The notebook's code cells are extracted verbatim from the written document, so the two cannot
drift apart.

## Quickstart (Colab)

1. Open `finetuning_tutorial.ipynb` in Google Colab.
2. **Runtime → Change runtime type → T4 GPU.** This matters: the notebook needs a GPU, and it
   detects that a T4 has no bfloat16 support and trains in fp16 instead.
3. Run the cells in order.

Rough timings on a free T4:

| Step | Time |
|---|---|
| Install libraries | 2–5 min |
| Download + quantize the 1.5B model | 1–3 min |
| Train (2,000 examples, 1 epoch) | 10–25 min |
| Evaluate + generate | 3–5 min |

Short on time? Drop the dataset slice from 2,000 to 500 examples in cell 4 — every concept
still applies.

## Running locally

Any NVIDIA GPU with ≥8 GB VRAM works. Install PyTorch for your CUDA version first (Colab
provides it already), then:

```bash
pip install -r requirements.txt
```

## A note on library versions

This ecosystem makes breaking changes every few months, and version drift is the single most
common reason a fine-tuning tutorial fails to run.

The important subtlety, covered in §7.2 of the tutorial: **"latest" is not the version you
want.** As of July 2026 the newest TRL on PyPI is 0.29.1, but Unsloth's package metadata
requires `trl<=0.24.0`; likewise `transformers` is at 5.14.1 while Unsloth supports `<=5.5.0`.
Installing the latest of everything produces a resolver conflict or a subtly broken
environment. The pins in cell 1 and in `requirements.txt` sit inside Unsloth's supported
window, cell 2 asserts the resolved versions, and cell 5 detects the installed API at runtime
so the notebook survives the next rename.

If you are reading this well after July 2026, re-derive the window rather than trusting the
pins:

```python
from importlib.metadata import requires
for r in requires("unsloth") or []:
    if r.split()[0].split("[")[0] in {"trl", "transformers", "peft", "torch"}:
        print(r)
```

## Licensing

- **Model** — Qwen2.5-1.5B-Instruct, released by Alibaba Cloud under a permissive open license
  (Apache-2.0 for most sizes in the family). Confirm on the model card before any use beyond
  coursework.
- **Dataset** — `yahma/alpaca-cleaned`. The Alpaca lineage was generated using OpenAI model
  outputs and inherits usage restrictions from that origin, typically non-commercial. Fine for
  coursework; use a cleanly licensed dataset for anything you ship.
- **Libraries** — Unsloth, `transformers`, `peft`, `trl` (Apache-2.0); `bitsandbytes` (MIT).

See §19 of the tutorial for the full attribution and reproducibility checklist.
