# TakeMeter — Planning Document
**AI201 Project 3 · Fine-Tuned Text Classifier · /r/nba**

---

## Label Taxonomy

*(To be defined — fill in before annotating)*

| Label | Definition | Key signal |
|---|---|---|
| | | |
| | | |
| | | |

**Design constraints:**
- Mutually exclusive: every post belongs to exactly one label
- Exhaustive: covers ≥90% of posts without a catch-all "other"
- Grounded in /r/nba community norms — the distinction must matter to real participants

---

## Dataset

- **Source:** /r/nba posts and/or comments
- **Target size:** ≥200 labeled examples
- **Split:** train / validation / test
- **Collection method:** *(to be documented)*
- **Label distribution:** *(fill in after labeling — aim for ≥20% per label)*

### Difficult examples (at least 3)

*(Document examples you found genuinely hard to label and what you decided)*

---

## Fine-Tuning Pipeline

- **Base model:** `distilbert-base-uncased`
- **Platform:** Google Colab (free T4 GPU)
- **Libraries:** `transformers`, `datasets`, `scikit-learn`
- **Training approach:** *(document after training)*
- **Key hyperparameter decisions:** *(learning rate, epochs, batch size)*

---

## Baseline Comparison

- **Baseline:** Zero-shot Groq `llama-3.3-70b-versatile`
- **Method:** Prompt Groq to classify each test example with no task-specific training
- **Groq prompt:** *(paste prompt here after writing it)*

---

## Evaluation Plan

Report on the held-out test set for both models:
- Overall accuracy (fine-tuned vs. baseline)
- Per-class: precision, recall, F1
- Confusion matrix
- At least 3 specific wrong predictions with analysis
- Reflection: what the model learned vs. what you intended

---

## Output Files (downloaded from Colab)

| File | Contents |
|---|---|
| `outputs/evaluation_results.json` | Full per-example predictions and metrics |
| `outputs/confusion_matrix.png` | Confusion matrix visualization |
