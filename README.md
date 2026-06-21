# TakeMeter
**AI201 Project 3 — Fine-Tuned Text Classifier · /r/nba**

TakeMeter classifies the discourse quality of posts and comments from the /r/nba community using a fine-tuned `distilbert-base-uncased` model, compared against a zero-shot Groq baseline.

---

## Label Taxonomy

*(To be filled in — see planning.md)*

---

## Dataset

- **Source:** /r/nba
- **Size:** ≥200 labeled examples
- **Split:** train / validation / test
- **Collection:** *(document where and how you collected the data)*
- **Label distribution:**

| Label | Count | % |
|---|---|---|
| | | |

### Labeling process

*(Describe your labeling process here)*

### Difficult examples

*(At least 3 examples you found genuinely hard to label, and what you decided)*

---

## Model

- **Base model:** `distilbert-base-uncased`
- **Fine-tuning platform:** Google Colab (T4 GPU)
- **Training approach:** *(describe)*
- **Key hyperparameter decisions:** *(learning rate, epochs, batch size — and why)*

---

## Baseline

Zero-shot classification with Groq `llama-3.3-70b-versatile`. No task-specific training — the prompt describes the labels and asks for a classification directly.

---

## Results

*(Fill in after running evaluation)*

| Model | Accuracy |
|---|---|
| Fine-tuned DistilBERT | |
| Zero-shot Groq baseline | |

See `outputs/evaluation_results.json` for full per-example results and `outputs/confusion_matrix.png` for the confusion matrix.

---

## Reflection

*(What did the model learn vs. what you intended it to learn?)*

---

## Repo Structure

```
ai201-project3-takemeter/
├── data/
│   └── labeled_data.csv       # annotated dataset
├── outputs/
│   ├── evaluation_results.json
│   └── confusion_matrix.png
├── planning.md
└── README.md
```
