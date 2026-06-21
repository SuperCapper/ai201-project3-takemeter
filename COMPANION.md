# TakeMeter — Companion Document
**AI201 Project 3 · Fine-Tuned Text Classifier · /r/nba**

---

## Overall Plan — Step by Step

| # | Step | Status |
|---|---|---|
| 1 | Design label taxonomy — 3 labels, definitions, edge case rules | ✅ Done |
| 2 | Collect 200+ text posts and comments from r/nba | ⬜ TODO |
| 3 | Annotate all examples; document 3 difficult cases | ⬜ TODO |
| 4 | Split into train / validation / test sets | ⬜ TODO |
| 5 | Upload CSV to Colab; configure label map in notebook | ⬜ TODO |
| 6 | Fine-tune `distilbert-base-uncased` on T4 GPU (~5–15 min) | ⬜ TODO |
| 7 | Write Groq zero-shot baseline prompt; run on test set | ⬜ TODO |
| 8 | Download `evaluation_results.json` + `confusion_matrix.png` from Colab | ⬜ TODO |
| 9 | Write evaluation report: accuracy, per-class F1, failure analysis | ⬜ TODO |
| 10 | Commit all outputs; fill in README and planning.md | ⬜ TODO |

---

## Label Taxonomy (finalized)

Three labels. The axis being measured: **how substantiated is the claim?**

| Label | One-sentence definition |
|---|---|
| `analysis` | The post makes a structured argument using specific, verifiable evidence — a stat, historical comparison, or tactical breakdown — where the evidence does genuine argumentative work. |
| `hot_take` | A bold, confident claim stated without supporting evidence, or with decorative evidence that doesn't genuinely support the argument; asserts rather than reasons. |
| `reaction` | An immediate emotional response to a specific, recent event (a game, play, trade, or news item); little to no argument; the post would not exist without that event. |

### Edge case rules

**`analysis` vs `hot_take` (hardest boundary):**
If removing the opinion framing leaves a verifiable, relevant argument standing → `analysis`.
If the stat is cherry-picked, one-dimensional, or selected for rhetorical effect rather than as part of a genuine argument → `hot_take`.
*Example: "LeBron is overrated — his playoff win rate vs. top seeds is below .500" → `hot_take`. One cherry-picked stat, accusatory framing, no alternative context.*

**`reaction` vs `hot_take`:**
If the claim could be written independently of the specific event that prompted it → `hot_take`, even if a game sparked it.
*Example: "After that game 7 choke, [player] has no killer mentality" → `hot_take`. The mental-toughness claim generalizes beyond last night.*

**`reaction` vs `analysis`:**
A detailed post-game tactical breakdown is `analysis` even if written about last night's game — deliberative structure overrides event-specificity.

### Scope decision
- **Text posts and comments only** — image posts, highlight links, and meme posts are excluded
- **Both top-level posts and comments** are included to ensure class balance (comments skew toward `reaction` and `hot_take`; top-level posts produce more `analysis`)

---

## Data: What We Have and What We Need

### Files on disk

| File | Contents | Status |
|---|---|---|
| `data/labeled_data.csv` | 200+ annotated examples — `id`, `source_type`, `text`, `label` | ⬜ Not yet collected |
| `outputs/evaluation_results.json` | Per-example predictions, ground truth, metrics (downloaded from Colab) | ⬜ Post fine-tuning |
| `outputs/confusion_matrix.png` | Confusion matrix visualization (downloaded from Colab) | ⬜ Post fine-tuning |

### CSV schema

```
id,source_type,text,label
rnba_001,post,"LeBron is clearly in decline...",hot_take
rnba_002,comment,"CURRY WITH THE 3 AT THE BUZZER",reaction
rnba_003,post,"Jokic's on/off differential is +9.2...",analysis
```

| Field | Type | Notes |
|---|---|---|
| `id` | string | Sequential: `rnba_001`, `rnba_002`, … |
| `source_type` | string | `post` or `comment` |
| `text` | string | Raw post/comment text; keep under ~512 tokens for DistilBERT |
| `label` | string | One of: `analysis`, `hot_take`, `reaction` |

### Target label distribution

Aim for ≥20% per label across the full 200+ examples.
Rough targets: `analysis` ~25%, `hot_take` ~40%, `reaction` ~35%
(hot_take and reaction are more common in the wild; analysis requires deliberate oversampling)

### Data collection approach

Reddit data — no API required for manual collection.
Optional: PRAW (Python Reddit API Wrapper) for bulk scraping.

**Manual collection strategy:**
- r/nba top posts of the week → text posts for `analysis` and `hot_take`
- r/nba game thread comments → comments for `reaction` and `hot_take`
- r/nba "Unpopular Opinion" and "Hot Takes" megathreads → `hot_take`
- r/nba post-game discussion threads → mix of all three

### API calls required

#### 1 — Groq zero-shot baseline (one call per test example)

```python
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "<taxonomy description>"},
        {"role": "user",   "content": f"Classify this post:\n\n{text}\n\nRespond with:\nLabel: <analysis|hot_take|reaction>\nReasoning: <one sentence>"},
    ],
    max_tokens=100,
)
label = response.choices[0].message.content
```

**Fields used:** `response.choices[0].message.content` — parsed for label string.
**Volume:** ~30 calls (one per test example). Free tier is sufficient.
**No streaming, no tool calls, no multi-turn.**

#### 2 — Hugging Face fine-tuning (Colab notebook — no API)

```python
# Tokenization
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
inputs = tokenizer(texts, truncation=True, padding=True, max_length=512)

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3
)

# Trainer
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./results", num_train_epochs=3, ...),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()
```

**No external API — all compute runs on Colab's free T4 GPU.**

---

## Transforms and Logic

### Preprocessing
1. Strip URLs, Reddit-specific formatting (`>` quotes, `**bold**`, subreddit links)
2. Collapse whitespace; trim leading/trailing space
3. Truncate to 512 tokens (DistilBERT's max context window)
4. Encode labels: `analysis=0`, `hot_take=1`, `reaction=2`

### Train / Validation / Test split
- Total: 200+ examples
- Split: 70% train (~140) / 15% validation (~30) / 15% test (~30)
- Stratified by label to maintain distribution across splits
- Test set is held out — never seen by the fine-tuned model during training

### Fine-tuning (Colab notebook)
- Base: `distilbert-base-uncased` (66M parameters, fast to fine-tune)
- Add classification head: linear layer → 3 output logits
- Loss: cross-entropy
- Key hyperparameters to document: learning rate, batch size, number of epochs
- Evaluation during training: validation loss + accuracy after each epoch

### Groq zero-shot baseline
- Same test set as fine-tuned model
- Prompt describes the 3 labels and requests `Label: X` / `Reasoning: Y` format
- Parse label by scanning for `Label:` prefix; validate against `{analysis, hot_take, reaction}`
- Invalid response → `unknown` (counted as incorrect)

### Evaluation metrics (both models)
- **Overall accuracy:** correct / total on test set
- **Per-class:** precision, recall, F1 for each of the 3 labels
- **Confusion matrix:** 3×3 table showing predicted vs. ground truth
- **Failure analysis:** at least 3 specific misclassified examples with written analysis

---

## Visualization — How Results Are Displayed

### Confusion matrix (`outputs/confusion_matrix.png`)
3×3 matrix — rows = ground truth, columns = predicted.
Generated by Colab notebook using `sklearn.metrics.ConfusionMatrixDisplay`.

### Evaluation results table (in README)
Side-by-side comparison of fine-tuned model vs. Groq baseline:

| Metric | Fine-tuned DistilBERT | Groq Zero-Shot |
|---|---|---|
| Overall accuracy | | |
| analysis — F1 | | |
| hot_take — F1 | | |
| reaction — F1 | | |

### Per-class breakdown (in README)
Precision / Recall / F1 for each label, both models.

### Failure analysis (in README)
At least 3 wrong predictions with:
- The post text
- Ground truth label
- Model's predicted label
- Written explanation of why the model may have erred

---

## What's Implemented / What's Left

### Done
- [x] Label taxonomy designed — 3 labels, one-sentence definitions, edge case rules for both hard boundaries
- [x] Project scaffold created — `data/`, `outputs/`, `planning.md`, `README.md`
- [x] GitHub repo initialized at `https://github.com/SuperCapper/ai201-project3-takemeter`
- [x] Scope decisions locked — text posts + comments; 3 labels; ~70/15/15 split

### Yet to build
- [ ] Collect 200+ posts/comments from r/nba
- [ ] Annotate all examples; document 3 difficult cases in README
- [ ] Build and upload `data/labeled_data.csv`
- [ ] Configure label map in Colab notebook
- [ ] Run fine-tuning on T4 GPU
- [ ] Write and run Groq zero-shot baseline
- [ ] Download and commit `evaluation_results.json` + `confusion_matrix.png`
- [ ] Fill in evaluation tables in README
- [ ] Write failure analysis and reflection

### Known risks / watch-outs
- **Class imbalance:** `hot_take` will be over-represented in the wild — actively seek `analysis` posts to avoid the model defaulting to `hot_take`
- **`analysis` vs `hot_take` boundary:** the edge case rule (decorative vs. genuine evidence) will require judgment on borderline posts — document the hard ones
- **Test set leakage:** keep test set in a separate variable/split from the start; never use it to guide annotation decisions
- **DistilBERT token limit:** posts over ~400 words will be truncated; note this in README

---

## Building Next and Why

**Next: collect and annotate the dataset**

The taxonomy is designed — the only thing blocking everything else is having 200+ labeled examples. Data collection and annotation is the critical path:

1. Browse r/nba and collect raw text into a spreadsheet or CSV (no code needed yet)
2. Read and label each example using the taxonomy above
3. Flag any examples that required a judgment call for the "difficult cases" section
4. Aim for at least 50 of each label before stopping — if one class is under 20%, go find more of that type specifically
5. Once the CSV is complete, upload to Colab and the rest of the pipeline follows

The hint from the assignment is worth repeating: *collect before you label, and read 30–40 first before committing to anything.* The taxonomy is already grounded in real post patterns, but annotation will surface edge cases the definitions don't yet cover.
