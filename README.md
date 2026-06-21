# TakeMeter
**AI201 Project 3 — Fine-Tuned Text Classifier · /r/nba**

TakeMeter classifies the discourse quality of posts and comments from the /r/nba community using a fine-tuned `distilbert-base-uncased` model, compared against a zero-shot Groq baseline.

---

## Label Taxonomy

Three labels. The central axis: **how substantiated is the claim?**

| Label | Definition |
|---|---|
| `analysis` | A structured argument using specific, verifiable evidence — a stat, historical comparison, or tactical breakdown — where the evidence does genuine argumentative work. |
| `hot_take` | A bold, confident claim stated without supporting evidence, or with decorative evidence that does not genuinely support the argument. Asserts rather than reasons; not tied to a single recent event. |
| `reaction` | An immediate emotional response to a specific recent event (a game, play, trade, or news item). Little to no argument; the post would not exist without that specific event. |

### Edge case rules

**`analysis` vs `hot_take`:** If removing the opinion framing leaves a verifiable, coherent argument standing → `analysis`. If the stat is cherry-picked or present to sound credible rather than to genuinely reason → `hot_take`. *Test: would a statistician call this evidence, or a rhetorical flourish?*

**`reaction` vs `hot_take`:** If the claim could be written independently of the specific event that prompted it → `hot_take`, even if a game sparked it. *Test: does the claim survive if you remove the game reference?*

---

## Dataset

- **Source:** /r/nba (AI-generated synthetic examples representative of real community discourse patterns — see AI usage section)
- **Size:** 220 labeled examples
- **Scope:** Text posts and comments only; image, highlight, and meme posts excluded
- **Split:** 70% train (154) / 15% validation (33) / 15% test (33), stratified by label
- **Label distribution:**

| Label | Count | % |
|---|---|---|
| `analysis` | 65 | 29.5% |
| `hot_take` | 90 | 40.9% |
| `reaction` | 65 | 29.5% |

### Labeling process

Each example was assigned a label using the taxonomy from `planning.md`. Examples were reviewed against the two edge case decision rules before finalizing labels. The first 50 examples were AI pre-labeled with the taxonomy in the prompt; all labels were reviewed and confirmed. The remaining 170 examples were labeled directly. The `suggested_label` column in `data/labeled_data.csv` identifies AI-pre-labeled rows.

### Difficult examples

| # | Post text (truncated) | Could be | Decision | Rule applied |
|---|---|---|---|---|
| 1 | "Karl Malone deserved the MVP award over Jordan in 1998... but Malone had a better regular season by every statistical measure." | `analysis` or `hot_take` | `hot_take` | "Better by every statistical measure" is asserted, not demonstrated — no specific stats cited, no peer comparison made. Removing the evidence framing leaves only an opinion. |
| 2 | "The refs decided that series... Every call went one way and anyone with eyes could see it." | `reaction` or `hot_take` | `hot_take` | References a specific series but "every call went one way" is a generalizing statement about the entire series, not a reaction to a single play. Survives removal of the series reference. |
| 3 | "The play-in loss hurts worse than a first-round exit... Getting knocked out before the bracket even starts feels uniquely embarrassing." | `reaction` or `hot_take` | `hot_take` | Likely prompted by a specific loss, but the claim is a general opinion about playoff formats. Does not describe the game, a play, or a specific moment. Generalizes completely from the triggering event. |

---

## Model

- **Base model:** `distilbert-base-uncased` (66M parameters)
- **Fine-tuning platform:** Google Colab (T4 GPU)
- **Training approach:** Classification head (linear → 3 logits) added to frozen base; fine-tuned end-to-end with cross-entropy loss
- **Key hyperparameters:**

| Hyperparameter | Value | Notes |
|---|---|---|
| Epochs | 3 | Default; sufficient for convergence on 154 examples |
| Learning rate | 2e-5 | Standard for DistilBERT fine-tuning |
| Batch size | 16 | Fits T4 memory comfortably |
| Max sequence length | 512 | DistilBERT's limit; all examples fit within this |

---

## Baseline

Zero-shot classification with Groq `llama-3.3-70b-versatile`. The prompt describes the three labels and edge case rules and requests a `Label: <label>` response. No labeled examples were provided.

**Baseline prompt (system message):**
```
You are classifying /r/nba posts and comments into one of three labels.

analysis: A structured argument using specific, verifiable evidence — a stat, historical comparison, or tactical breakdown — where the evidence does genuine argumentative work.
hot_take: A bold, confident claim stated without supporting evidence, or with decorative evidence that does not genuinely support the argument. Asserts rather than reasons.
reaction: An immediate emotional response to a specific recent event. Little to no argument; the post would not exist without that specific event.

Respond with exactly:
Label: <analysis|hot_take|reaction>
Reasoning: <one sentence>
```

---

## Results

### Summary

| Metric | Fine-tuned DistilBERT | Groq Zero-Shot | Improvement |
|---|---|---|---|
| Overall accuracy | **84.8%** | 66.7% | +18.2 pp |
| Macro-averaged F1 | **0.852** | 0.666 | +0.186 |
| `analysis` F1 | **0.842** | 0.632 | +0.210 |
| `hot_take` F1 | **0.815** | 0.667 | +0.148 |
| `reaction` F1 | **0.900** | 0.700 | +0.200 |

### Per-class breakdown — Fine-tuned DistilBERT

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `analysis` | 0.889 | 0.800 | 0.842 | 10 |
| `hot_take` | 0.786 | 0.846 | 0.815 | 13 |
| `reaction` | 0.900 | 0.900 | 0.900 | 10 |
| **macro avg** | **0.858** | **0.849** | **0.852** | **33** |

### Per-class breakdown — Groq Zero-Shot Baseline

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `analysis` | 0.667 | 0.600 | 0.632 | 10 |
| `hot_take` | 0.643 | 0.692 | 0.667 | 13 |
| `reaction` | 0.700 | 0.700 | 0.700 | 10 |
| **macro avg** | **0.670** | **0.664** | **0.666** | **33** |

### Confusion matrix — Fine-tuned DistilBERT

*(See `outputs/confusion_matrix.png` for the visual. Text version:)*

| True \ Predicted | analysis | hot\_take | reaction |
|---|---|---|---|
| **analysis** | **8** | 2 | 0 |
| **hot\_take** | 1 | **11** | 1 |
| **reaction** | 0 | 1 | **9** |

### Confusion matrix — Groq Zero-Shot Baseline

| True \ Predicted | analysis | hot\_take | reaction |
|---|---|---|---|
| **analysis** | **6** | 3 | 1 |
| **hot\_take** | 2 | **9** | 2 |
| **reaction** | 1 | 2 | **7** |

### Success criteria check

| Criterion | Threshold | Fine-tuned result | Pass? |
|---|---|---|---|
| Overall accuracy | ≥ 70% | 84.8% | ✅ |
| Macro-averaged F1 | ≥ 0.65 | 0.852 | ✅ |
| `analysis` F1 | ≥ 0.60 | 0.842 | ✅ |
| `hot_take` F1 | ≥ 0.65 | 0.815 | ✅ |
| `reaction` F1 | ≥ 0.65 | 0.900 | ✅ |
| Fine-tuned vs. baseline (macro-F1 gap) | ≥ +0.05 | +0.186 | ✅ |

---

## Failure Analysis

5 of 33 test examples were misclassified by the fine-tuned model. Three are analyzed in depth below.

### Error 1 — `analysis` predicted as `hot_take` (rnba_019)

**Post text:**
> "Kyrie's shot chart shows an extremely high volume of contested mid-range and pull-up attempts that most shot-selection models mark as inefficient. His actual efficiency on those shots is in the 95th percentile for the positions he takes them from — which is either extraordinary skill or a massive sample-size anomaly."

**True label:** `analysis` — **Predicted:** `hot_take`

**Analysis:** The post references shot-selection models and 95th-percentile efficiency, which should signal analysis. However, the framing is explicitly uncertain ("either extraordinary skill or a massive sample-size anomaly"), which may have read as hedged opinion rather than a structured argument. The model likely learned that analysis posts assert a conclusion supported by evidence; this one presents evidence and explicitly declines to conclude. That is actually a more rigorous form of analysis, but the model may not have learned to recognize it. Additionally, the phrase "most shot-selection models mark as inefficient" implies the post is countering conventional wisdom — a contrarian stance that can look like a hot take in surface structure even when backed by data.

### Error 2 — `hot_take` predicted as `analysis` (rnba_127)

**Post text:**
> "Playoff basketball is when we find out who the real players are, and Harden has failed that test every single time it counted. Great numbers during the regular season. Historically bad when his team needed him most."

**True label:** `hot_take` — **Predicted:** `analysis`

**Analysis:** The word "historically" is almost certainly responsible for this error. The model likely learned that words like "historically," "statistically," and "data" correlate with analysis labels. But "historically bad" here is rhetorical — the post does not cite historical data, it asserts a characterization. This is exactly the decorative-evidence pattern described in the edge case rules (§3 of planning.md): language that sounds like evidence but does not do genuine argumentative work. The model has learned surface features of analytical vocabulary but not the underlying distinction between genuine evidence and credibility-signaling language.

### Error 3 — `reaction` predicted as `hot_take` (rnba_197)

**Post text:**
> "Whatever happens in this series, tonight's performance from Tyrese Haliburton is the best I have seen from him. Ice in his veins in the final four minutes. He looked like a different player."

**True label:** `reaction` — **Predicted:** `hot_take`

**Analysis:** This was also a documented difficult case during annotation (see planning.md §3). The post was written in response to a specific game performance, but contains two generalizing claims: "the best I have seen from him" (across his career) and "he looked like a different player" (a character judgment, not just a game reaction). Both of these survive removal of the game reference — you could write "Haliburton looks like a different player this season" independently of tonight's game. The model correctly identified the generalizing language but failed to weight the fact that the primary content is event-specific and time-bound. This is a genuine boundary case where a reasonable annotator could disagree.

---

## Reflection

**What the model learned vs. what I intended it to learn:**

The model learned the surface linguistic features of each class very effectively. `reaction` was the easiest class (F1 = 0.900) because the vocabulary and punctuation patterns are highly distinctive — ALL CAPS text, exclamations, event-specific names and pronouns ("they," "he just," "right now") are strong signals. `analysis` was harder (F1 = 0.842) because the line between analytical language and analytical-sounding hot takes is subtle.

The failure pattern in Errors 1 and 2 reveals the same underlying limitation: the model learned vocabulary correlation rather than argumentative structure. It associated "historically," "statistically," and explicit uncertainty hedges with a specific class, but it has not learned to ask whether the evidence actually supports the claim. That distinction — evidence that does genuine argumentative work vs. evidence that sounds credible — was the hardest thing to operationalize in the taxonomy, and the model struggled with it in the same places a human annotator would struggle.

The +18.6 macro-F1 improvement over the zero-shot baseline confirms that fine-tuning on even 154 examples with clear labels produces a meaningfully better classifier. The baseline's errors were more distributed — it confused all three classes in multiple directions — while the fine-tuned model's errors concentrate on the `analysis`/`hot_take` boundary, which is exactly where the hardest examples live.

**What I would do with more data:**

The biggest gain would come from more `analysis` examples that contain statistical language but are still `hot_take` — so the model is forced to learn the distinction from evidence that actually works vs. evidence that is decorative. That is a harder training signal than the current dataset provides, where `hot_take` examples mostly don't contain statistics at all.

---

## AI Usage Disclosure

- **Dataset generation:** All 220 training examples are AI-generated synthetic posts representative of r/nba discourse patterns. Direct Reddit scraping was unavailable; ChatGPT-3.5 and Claude Sonnet 4.6 were used to generate examples across all three label categories.
- **Pre-labeling:** The first 50 examples were pre-labeled using the taxonomy as an LLM prompt; all labels were reviewed and confirmed before training. The `suggested_label` column in the CSV identifies these rows.
- **Failure analysis:** The failure pattern descriptions above were developed through direct reading of wrong predictions. No LLM clustering was used for this analysis.

---

## Repo Structure

```
ai201-project3-takemeter/
├── data/
│   └── labeled_data.csv         # 220 annotated examples
├── outputs/
│   ├── evaluation_results.json  # per-example predictions and aggregate metrics
│   └── confusion_matrix.png     # side-by-side confusion matrices (both models)
├── planning.md                  # taxonomy, metrics, success criteria, AI tool plan
├── COMPANION.md                 # pipeline documentation and build log
└── README.md
```
