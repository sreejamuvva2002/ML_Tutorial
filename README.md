# Fine-Tuning Large Language Models: LoRA and QLoRA

A hands-on tutorial that fine-tunes **Qwen2.5-1.5B-Instruct** with **QLoRA** — a frozen 4-bit base
model with small trainable low-rank adapters on top — on a mixture of general instruction data and
a deterministic structured-extraction task, and evaluates the result against its own un-tuned
baseline on both. Runs end to end on a **single free Google Colab T4 GPU (16 GB)**. No paid
compute, no API keys, no private data.

Course tutorial submission — ARTI 4555/6555, MS in Artificial Intelligence.
Author: Sreeja Muvva.

## Contents

| File | Purpose |
|---|---|
| [`Fine_Tuning_LLMs_Tutorial.md`](Fine_Tuning_LLMs_Tutorial.md) | The written tutorial: concepts, code, interpretation, exercises |
| [`finetuning_tutorial.ipynb`](finetuning_tutorial.ipynb) | The runnable experiment |
| [`finetuning_tutorial_executed_run2.ipynb`](finetuning_tutorial_executed_run2.ipynb) | The final run on an RTX A5000, fully executed — source of `results/` |
| [`finetuning_tutorial_executed_run2_t4.ipynb`](finetuning_tutorial_executed_run2_t4.ipynb) | The same notebook re-executed on a free Colab T4 (replication) |
| [`finetuning_tutorial_executed_run1.ipynb`](finetuning_tutorial_executed_run1.ipynb) | The earlier Alpaca-only run (the negative result) |
| `results/` | `metrics.json`, `loss_curves.png`, `environment.json`, and per-example prediction logs |
| `requirements.txt` | Pinned dependencies for local (non-Colab) runs |

The written tutorial's §8 is a minimal teaching path; the notebook implements the stricter
experiment the results come from. §8 states the differences explicitly.

## Results

Two runs, and the pair is the finding. Full write-up in §11.5–§11.6.

### Run 1 — general instruction data only (Tesla T4)

Fine-tuned on Alpaca alone. Peak VRAM 4.06 GB, 9.4 min, best validation loss 1.0211.

**The result was negative.** Across five prompts the tuned model was never clearly better and was
worse on two, most visibly turning `- Milk` into `- We need milk.` Qwen2.5-1.5B-**Instruct** is
already instruction-tuned and Alpaca is distilled from a weaker teacher, so SFT taught a strong
model to imitate a worse one. Validation loss fell while human-visible quality did not improve.

### Run 2 — structured task with general replay (RTX A5000, replicated on a T4)

Changed the experiment rather than the hyperparameters: added a deterministic company-record → JSON
task with exact field-level gold answers, mixed with general Alpaca data, and evaluated **both**
model conditions on untouched test splits. 5.1 min, peak 2.19 GB, validation loss 0.8013. Re-executed on a free Colab T4 in fp16: 8.9 min, peak 2.86 GB, validation loss 0.8014 — every conclusion replicates (§11.5).

**In-distribution structured test (n=60)**

| Metric | Base | Tuned |
|---|---|---|
| Strict JSON-only rate | 0.000 | **1.000** |
| All fields exactly correct | 0.750 | **1.000** |
| City accuracy | 0.800 | **1.000** |

**Surface-form challenge set (n=48)** — same task and schema, six unseen record templates, unseen
cities and company names, distractor sentences, unfamiliar number and certification phrasings.

| Metric | Base | Tuned |
|---|---|---|
| All fields exactly correct | 0.438 | **0.562** |
| Supply-chain role accuracy | 0.458 | 0.583 |
| Strict JSON-only rate | 0.000 | **1.000** |

**Held-out general instruction data (ROUGE, n=100):** ROUGE-1 0.4115 → 0.4549, ROUGE-2 0.1787 →
0.2170, ROUGE-L 0.2857 → 0.3326. No regression.

### What this does and does not show

Fine-tuning did nothing when there was no clearly defined missing capability, and produced a large,
exactly-measurable gain once there was one. But read the challenge set before concluding the task
was solved:

- **A perfect in-distribution score is not a solved task.** The same adapter scores 1.000 on
  records drawn from the training templates and 0.562 on records whose *shape* it has not seen.
- **Part of that drop is the task, not the adapter.** The base model also falls, 0.750 → 0.438, so
  only the excess of the tuned gap over the base gap (0.4375 − 0.3125 = 0.125) is attributable to
  template dependence.
- **The challenge gain is concentrated in one format.** Per-template, the entire improvement comes
  from the Q&A layout (0.125 → 0.875); the bullet-list format scores 0.000 for both models.
- **Almost every remaining failure is one field.** 20 of 21 tuned challenge failures are
  `supply_chain_role` — a closed-set label, unlike the literal spans the model extracts perfectly.
- **The intervals overlap.** 95% Wilson at n=48: base [0.307, 0.577], tuned [0.423, 0.693]. The
  +0.125 out-of-distribution gain is suggestive, not established.
- ROUGE stability shows no degradation appeared on the sampled data — not that general capability
  was preserved. The replay ablation needed to establish causation has not been run.

## Quickstart (Colab)

1. Open `finetuning_tutorial.ipynb` in Google Colab.
2. **Runtime → Change runtime type → T4 GPU.** The notebook detects that a T4 has no hardware
   bfloat16 and selects fp16 — see the precision trap in §7.4, which is subtler than it looks.
3. Run every cell in order from a fresh runtime.

Expect roughly 35–45 minutes: about 5–9 minutes of training, and the rest generation, since every
evaluation prompt is generated twice, once per model condition.

To shorten it, lower `ALPACA_EXAMPLES` and `STRUCTURED_EXAMPLES` in the configuration cell — every
concept still applies.

## Running locally

Any NVIDIA GPU with ≥8 GB VRAM works. Install PyTorch for your CUDA version first (Colab provides
it already), then:

```bash
pip install -r requirements.txt
```

The final code measured 2.86 GB on a T4 and 2.19 GB on an A5000. An earlier build measured 6.17 GB
on the same T4 purely because `unsloth` was imported after `transformers` — see §6.2, where that
one-line ordering is worth more than 3 GB.

## A note on library versions

Version drift is the single most common reason a fine-tuning tutorial fails to run.

The subtlety, covered in §7.2: **"latest" is not the version you want.** As of July 2026 the newest
TRL on PyPI is 0.29.1, but Unsloth's metadata requires `trl<=0.24.0`; likewise `transformers` is at
5.14.1 while Unsloth supports `<=5.5.0`. Installing the latest of everything produces a resolver
conflict or a subtly broken environment. The notebook pins inside Unsloth's supported window and
asserts the resolved versions at startup.

If you are reading this well after July 2026, re-derive the window rather than trusting the pins:

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
- **Dataset** — `yahma/alpaca-cleaned`. The Alpaca lineage was generated using OpenAI model outputs
  and inherits usage restrictions from that origin, typically non-commercial. Fine for coursework;
  use a cleanly licensed dataset for anything you ship.
- **Libraries** — Unsloth, `transformers`, `peft`, `trl` (Apache-2.0); `bitsandbytes` (MIT).

See §19 of the tutorial for the full attribution and reproducibility checklist.
