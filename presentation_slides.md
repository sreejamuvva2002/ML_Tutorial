# Presentation deck — content for each slide
*Fine-Tuning LLMs with LoRA and QLoRA — ARTI 4555/6555 · Sreeja Muvva*

Paste this into Claude and ask it to design a ~14-slide deck. Each slide below gives a **title**,
**on-slide content** (keep it terse — these are talking points, not paragraphs), a **visual**
suggestion, and **speaker notes** (what to say aloud). Target length: 10–12 minutes.

---

## Slide 1 — Title
**Fine-Tuning Large Language Models: LoRA & QLoRA**
- A hands-on tutorial — concepts, a real experiment, and honest results
- Sreeja Muvva · ARTI 4555/6555 · MS in Artificial Intelligence
- Model: Qwen2.5-1.5B-Instruct · runs on a free Colab T4 (16 GB)

**Visual:** clean title slide; small chip: "trained in 5–9 min, <3 GB VRAM."
**Notes:** One-line hook — "You can adapt a 1.5-billion-parameter model on a free GPU in minutes, and I'll show you when that's worth doing and when it isn't."

---

## Slide 2 — The problem fine-tuning solves
- Base instruct models are general; sometimes you need a **specific behaviour or output contract**
- Example: turn a messy company record into **strict JSON** with exact keys, every time
- Prompting can get you close; fine-tuning makes the behaviour **reliable**

**Visual:** left = a base model reply wrapped in ```json fences + prose; right = clean bare JSON.
**Notes:** Motivate with the concrete task the experiment uses. Reliability/format compliance is the win, not new knowledge.

---

## Slide 3 — When NOT to fine-tune (the most useful minute)
- **Need fresh/large/changing facts?** → use **retrieval (RAG)**, not fine-tuning
- **Need a behaviour, format, tone, or task pattern?** → fine-tuning fits
- Fine-tuning teaches *how to respond*, not *what is true*

**Visual:** 2-column decision table (RAG vs fine-tune) — reuse the §3 table.
**Notes:** Emphasise this is the slide that saves the audience the most time/money. Fine-tuning is not a knowledge-injection tool.

---

## Slide 4 — LoRA: the core idea
- Freeze the base weights **W**; learn a small low-rank update **ΔW = B·A**
- Only **A** and **B** train; **W** never moves
- Tiny, swappable adapters — base model stays intact (reduces forgetting risk)

**Visual:** diagram — frozen W (large) + B·A (small) → W + ΔW.
**Notes:** Rank r is the knob: small r = few params. Adapter can be toggled on/off over the same weights.

---

## Slide 5 — The parameter arithmetic (why it's cheap)
- Rank 16, applied to attention + MLP projections
- **Trainable: 18,464,768 params = 1.18%** of 1.56 B total
- You update ~1 in 85 parameters and still shift behaviour measurably

**Visual:** big stat tiles — "18.46 M trainable", "1.18% of model", "0 base weights changed".
**Notes:** This number matches the code's `print_trainable_parameters()` exactly — verified, not estimated.

---

## Slide 6 — QLoRA: fitting it on a free GPU
- Load the frozen base in **4-bit (NF4)**; train the adapters in higher precision
- Gradients flow *through* the quantised weights to the adapters
- Result: **peak 2.19 GB VRAM (A5000) / 2.86 GB (T4)** — fits 16 GB with huge headroom

**Visual:** memory bar — full fine-tune (won't fit) vs QLoRA (~2–3 GB) against a 16 GB line.
**Notes:** "Q" = quantised base. This is what makes it classroom-runnable. Peak VRAM measured, not predicted.

---

## Slide 7 — The experiment design
- **Two training signals:** general instructions (Alpaca) + a deterministic **company-record → JSON** task
- Evaluate **base vs. tuned** (adapter off vs. on) — same weights, fair comparison
- Held-out test split + a **surface-form challenge set** (unseen templates) to catch memorisation

**Visual:** pipeline diagram: data → QLoRA train → evaluate (base vs tuned) on 3 suites.
**Notes:** The base-vs-tuned toggle and the challenge set are what make the results trustworthy.

---

## Slide 8 — Setup & the version trap
- Pinned, tested stack: **unsloth 2026.7.5, trl 0.24.0, transformers 5.5.0, peft 0.19.1**
- "Latest" ≠ compatible: newer TRL (now 1.x) sits **outside** Unsloth's supported window
- Import **unsloth before transformers**; pick bf16/fp16 by GPU capability

**Visual:** callout box: "latest ≠ what you want — pin inside the framework's window."
**Notes:** Real lesson from the build: import ordering alone changed peak VRAM by >3 GB. Version drift is the #1 reason these tutorials fail to run.

---

## Slide 9 — Training results
- **1 epoch, 1,600 examples, ~5.1 min** (A5000), 200 optimizer steps
- Validation loss **0.807 → 0.801**, perplexity **2.23**; best checkpoint restored
- Loss curve is the honest signal — task metrics are where the win shows

**Visual:** the loss curve — `results/a5000/loss_curves.png`.
**Notes:** Validation moves little in one epoch on a strong instruct model; don't oversell the loss delta — point to the next slides.

---

## Slide 10 — Results: the format contract is learned
In-distribution structured test (n = 60):

| Metric | Base | Tuned |
|---|---|---|
| Strict JSON-only | 0.000 | **1.000** |
| All fields exact | 0.750 | **1.000** |
| City accuracy | 0.800 | **1.000** |

**Visual:** before/after — base reply in ```json fences + prose vs. tuned bare JSON.
**Notes:** Headline: base *never* returns bare JSON; tuned *always* does. That 0 → 1.000 is the single clearest effect.

---

## Slide 11 — Results: no regression on general ability
- Held-out Alpaca ROUGE (n = 100): **R1 0.412 → 0.455**, R2 0.179 → 0.217, RL 0.286 → 0.333
- Every metric up → the JSON training did **not** degrade general instruction-following
- Claim scoped: "no regression on sampled data," not "capability preserved"

**Visual:** grouped bar chart, base vs tuned, three ROUGE metrics.
**Notes:** Modest but positive; ROUGE is surface overlap only. Honesty about scope is the point.

---

## Slide 12 — The challenge set: don't overclaim
- Same task, **6 unseen record templates**, unseen cities/companies, distractors
- All-fields: in-distribution **1.000** → challenge **0.562** (A5000)
- Base also drops (0.750 → 0.438), so only the **excess gap (~0.125)** is adapter template-dependence
- Gain is concentrated in **one template** (Q&A 0.125 → 0.875); **20 of 21 failures = one field** (`supply_chain_role`)

**Visual:** two bars (in-dist vs challenge) for base and tuned + a small per-template strip.
**Notes:** This slide is the integrity of the whole project — a perfect in-distribution score is not a solved task. Report the base arm and the intervals.

---

## Slide 13 — Cross-environment replication (robustness)
- Ran the experiment on **A5000/bf16** and again on a free **T4/fp16** (the shipped loader)
- Different GPU, precision, Python/torch, and model-loading path — **same conclusions**
- Val loss 0.8013 vs 0.8014; strict-JSON 0→1 both; challenge tuned 0.562 vs 0.542 (1 record, inside the intervals)

**Visual:** side-by-side A5000 vs T4 mini-table (reuse §11.5 replication table).
**Notes:** Cross-hardware replication is the cheapest real robustness check a student project can offer — conclusions that survive a hardware+precision swap.

---

## Slide 14 — Pitfalls & what I'd do differently
- **"Latest" versions** break the install — pin inside the framework's window
- **Evaluate both conditions** and on **held-out + out-of-distribution** data, or you'll overclaim
- **Report uncertainty** (Wilson intervals overlap here) and a noise floor for small deltas
- Next: a replay ablation to test whether mixing general data prevented forgetting

**Visual:** simple checklist / "gotchas" list.
**Notes:** Close on rigor: the negative Run-1 result and the challenge set are features, not failures.

---

## Slide 15 — Takeaways
- QLoRA adapts a 1.5 B model on a **free GPU in minutes** (~1.2% params, <3 GB)
- Fine-tuning reliably taught a **format contract** (0 → 100% strict JSON); no regression on general ability
- Honest evaluation (base arm + challenge set + intervals + cross-env replication) matters more than any single number
- Repo: tutorial, runnable notebook, 3 executed runs, full results, MIT/CC-BY licensed

**Visual:** 3–4 takeaway tiles; QR or repo URL optional.
**Notes:** End on the method, not the metric — "the discipline of the evaluation is the real deliverable."

---

### Design guidance for Claude
- Consistent theme; one idea per slide; large type; minimal text per bullet.
- Use the two real images: `results/a5000/loss_curves.png` (Slide 9) and, if desired, `results/t4/loss_curves.png`.
- Charts: base vs tuned as paired bars; keep the base/tuned colour mapping consistent across slides.
- Keep the before/after JSON example (base fenced vs tuned bare) as a recurring visual motif.
