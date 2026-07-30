"""Dataset statistics for the report's Data Summarization section.

The report rubric asks for a statistical summary of the data. Nothing in the repository
computed one, so this does.

HOW TO RUN
----------
Paste the whole file into a new cell in `finetuning_tutorial.ipynb`, placed **after** the
dataset-construction cell and the challenge-set cell (it needs `raw_alpaca_*`,
`raw_structured_*`, `challenge_ds`, `build_alpaca_prompt`, and `tokenizer`). It needs no
GPU and no retraining — about two minutes on any runtime.

Writes `results/data_stats.json` and two figures under `results/`. Every number the paper's
Data section cites comes from that JSON, so the text and the data cannot drift.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev

import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return float(sorted_values[idx])


def describe(values: list[int]) -> dict[str, float]:
    """Five-number-ish summary. Population sd — this is the whole corpus, not a sample."""
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": round(mean(values), 1),
        "median": round(median(values), 1),
        "sd": round(pstdev(values), 1) if len(values) > 1 else 0.0,
        "min": ordered[0],
        "p95": _quantile(ordered, 0.95),
        "max": ordered[-1],
    }


def token_len(text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])  # noqa: F821


# --------------------------------------------------------------------- lengths
def alpaca_lengths(split) -> dict[str, dict]:
    prompts = [build_alpaca_prompt(ex) for ex in split]      # noqa: F821
    completions = [str(ex["output"]).strip() for ex in split]
    p = [token_len(t) for t in prompts]
    c = [token_len(t) for t in completions]
    return {
        "prompt_tokens": describe(p),
        "completion_tokens": describe(c),
        "total_tokens": describe([a + b for a, b in zip(p, c)]),
        "over_max_seq_len": sum(1 for a, b in zip(p, c) if a + b > MAX_SEQ_LEN),  # noqa: F821
    }


def structured_lengths(split) -> dict[str, dict]:
    p = [token_len(str(ex["prompt_text"])) for ex in split]
    c = [token_len(str(ex["gold_json"])) for ex in split]
    return {
        "prompt_tokens": describe(p),
        "completion_tokens": describe(c),
        "total_tokens": describe([a + b for a, b in zip(p, c)]),
        "over_max_seq_len": sum(1 for a, b in zip(p, c) if a + b > MAX_SEQ_LEN),  # noqa: F821
    }


# ------------------------------------------------------------------- balances
def class_balance(split, field: str) -> dict[str, float]:
    counts = Counter(str(ex[field]) for ex in split)
    total = sum(counts.values())
    return {k: round(v / total, 4) for k, v in sorted(counts.items())}


def duplicate_report(named_splits: dict[str, list[str]]) -> dict:
    """Exact-duplicate counts within and across splits — leakage evidence for the paper."""
    report = {"within": {}, "across": {}}
    for name, texts in named_splits.items():
        report["within"][name] = len(texts) - len(set(texts))
    names = list(named_splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(named_splits[a]) & set(named_splits[b])
            report["across"][f"{a} vs {b}"] = len(overlap)
    return report


stats: dict = {"max_seq_len": MAX_SEQ_LEN, "seed": SEED}  # noqa: F821

# ------------------------------------------------------------------- Alpaca
stats["alpaca"] = {
    "source": DATASET_ID,                       # noqa: F821
    "revision": resolved_dataset_revision,      # noqa: F821
    "splits": {
        "train": len(raw_alpaca_train),         # noqa: F821
        "validation": len(raw_alpaca_validation),  # noqa: F821
        "test": len(raw_alpaca_test),           # noqa: F821
    },
    "lengths": {
        "train": alpaca_lengths(raw_alpaca_train),            # noqa: F821
        "validation": alpaca_lengths(raw_alpaca_validation),  # noqa: F821
        "test": alpaca_lengths(raw_alpaca_test),              # noqa: F821
    },
}

# The optional `input` field is the main structural variation in Alpaca rows.
for split_name, split in [("train", raw_alpaca_train),                 # noqa: F821
                          ("validation", raw_alpaca_validation),        # noqa: F821
                          ("test", raw_alpaca_test)]:                   # noqa: F821
    has_input = sum(1 for ex in split if str(ex.get("input", "") or "").strip())
    stats["alpaca"].setdefault("with_input_fraction", {})[split_name] = round(
        has_input / len(split), 4
    )

# Leading instruction verb — a cheap description of task mix.
verbs = Counter(
    re.split(r"[^A-Za-z]+", str(ex["instruction"]).strip().lower())[0]
    for ex in raw_alpaca_train                                          # noqa: F821
    if str(ex["instruction"]).strip()
)
stats["alpaca"]["top_instruction_verbs"] = dict(verbs.most_common(15))

# ---------------------------------------------------------------- structured
stats["structured"] = {
    "generation": "synthetic, deterministic; 4 record templates x 12 cities x 5 roles",
    "splits": {
        "train": len(raw_structured_train),         # noqa: F821
        "validation": len(raw_structured_validation),  # noqa: F821
        "test": len(raw_structured_test),           # noqa: F821
    },
    "lengths": {
        "train": structured_lengths(raw_structured_train),            # noqa: F821
        "validation": structured_lengths(raw_structured_validation),  # noqa: F821
        "test": structured_lengths(raw_structured_test),              # noqa: F821
    },
    "balance": {
        field: {
            "train": class_balance(raw_structured_train, field),        # noqa: F821
            "test": class_balance(raw_structured_test, field),          # noqa: F821
        }
        for field in ("supply_chain_role", "iso_9001", "city")
    },
    "employee_count": {
        "train": describe([int(ex["employee_count"]) for ex in raw_structured_train]),  # noqa: F821
        "test": describe([int(ex["employee_count"]) for ex in raw_structured_test]),    # noqa: F821
    },
}

# ----------------------------------------------------------------- challenge
stats["challenge"] = {
    "n": len(challenge_ds),                       # noqa: F821
    "lengths": structured_lengths(challenge_ds),  # noqa: F821
    "balance": {
        field: class_balance(challenge_ds, field)  # noqa: F821
        for field in ("supply_chain_role", "iso_9001", "template_name", "headcount_style")
    },
    "disjointness": {
        "company_names_shared_with_training": len(
            set(structured_all["company_name"]) & set(challenge_ds["company_name"])  # noqa: F821
        ),
        "cities_shared_with_training": len(set(CITIES) & set(CHALLENGE_CITIES)),  # noqa: F821
        "prompts_shared_with_training": len(
            set(raw_structured_train["prompt_text"]) & set(challenge_ds["prompt_text"])  # noqa: F821
        ),
    },
}

# ---------------------------------------------------------------- duplicates
stats["duplicates"] = {
    "alpaca": duplicate_report({
        "train": [build_alpaca_prompt(ex) for ex in raw_alpaca_train],            # noqa: F821
        "validation": [build_alpaca_prompt(ex) for ex in raw_alpaca_validation],  # noqa: F821
        "test": [build_alpaca_prompt(ex) for ex in raw_alpaca_test],              # noqa: F821
    }),
    "structured": duplicate_report({
        "train": list(raw_structured_train["prompt_text"]),        # noqa: F821
        "validation": list(raw_structured_validation["prompt_text"]),  # noqa: F821
        "test": list(raw_structured_test["prompt_text"]),          # noqa: F821
        "challenge": list(challenge_ds["prompt_text"]),            # noqa: F821
    }),
}

with (RESULTS_DIR / "data_stats.json").open("w", encoding="utf-8") as handle:
    json.dump(stats, handle, indent=2)

# -------------------------------------------------------------------- figures
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

alp = [token_len(build_alpaca_prompt(ex)) + token_len(str(ex["output"]).strip())  # noqa: F821
       for ex in raw_alpaca_train]                                                # noqa: F821
strc = [token_len(str(ex["prompt_text"])) + token_len(str(ex["gold_json"]))
        for ex in raw_structured_train]                                           # noqa: F821

axes[0].hist(alp, bins=40)
axes[0].axvline(MAX_SEQ_LEN, linestyle="--", label=f"max_length={MAX_SEQ_LEN}")  # noqa: F821
axes[0].set_title("Alpaca train: total tokens per example")
axes[0].set_xlabel("tokens"); axes[0].set_ylabel("examples"); axes[0].legend()

axes[1].hist(strc, bins=40)
axes[1].axvline(MAX_SEQ_LEN, linestyle="--", label=f"max_length={MAX_SEQ_LEN}")  # noqa: F821
axes[1].set_title("Structured train: total tokens per example")
axes[1].set_xlabel("tokens"); axes[1].legend()

fig.tight_layout()
fig.savefig(RESULTS_DIR / "length_distributions.png", dpi=160)
plt.show()

print(json.dumps(stats, indent=2)[:4000])
print("\nWrote results/data_stats.json and results/length_distributions.png")
