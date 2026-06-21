# TakeMeter
**AI201 Project 3 — Fine-Tuned Text Classifier · /r/nba**

TakeMeter classifies the discourse quality of posts and comments from the /r/nba community using a fine-tuned `distilbert-base-uncased` model, compared against a zero-shot Groq baseline. The classifier labels each post as `analysis`, `hot_take`, or `reaction` — measuring how substantiated a claim is rather than just what it is about.

---

## Label Taxonomy

Three labels. The central axis: **how substantiated is the claim?**

| Label | Definition |
|---|---|
| `analysis` | A structured argument using specific, verifiable evidence — a stat, historical comparison, or tactical breakdown — where the evidence does genuine argumentative work. A reader could check the data; removing the evidence would weaken the argument. |
| `hot_take` | A bold, confident claim stated without supporting evidence, or with decorative evidence that does not genuinely support the argument. Asserts rather than reasons; claims are general and not tied to a single recent event. |
| `reaction` | An immediate emotional response to a specific recent event (a game, play, trade, or news item). Little to no argument; the post would not exist without that specific event. Often time-sensitive. |

### Edge case rules

**`analysis` vs `hot_take`:** If removing the opinion framing leaves a verifiable, coherent argument standing → `analysis`. If the stat is cherry-picked or present to sound credible rather than to genuinely reason → `hot_take`. *Test: would a statistician call this evidence, or a rhetorical flourish?*

**`reaction` vs `hot_take`:** If the claim could be written independently of the specific event that prompted it → `hot_take`, even if a game sparked it. *Test: does the claim survive if you remove the game reference?*

**`reaction` vs `analysis`:** A detailed post-game tactical breakdown is `analysis` even if written about last night's game — deliberative structure overrides event-specificity.

---

## Dataset

- **Source:** /r/nba (AI-generated synthetic examples representative of real community discourse patterns — see AI Usage section)
- **Size:** 220 labeled examples
- **Scope:** Text posts and comments only; image, highlight, link, and meme posts excluded
- **Split:** 70% train (154) / 15% validation (33) / 15% test (33), stratified by label — handled by the Colab notebook
- **Label distribution:**

| Label | Count | % of dataset |
|---|---|---|
| `analysis` | 65 | 29.5% |
| `hot_take` | 90 | 40.9% |
| `reaction` | 65 | 29.5% |

`hot_take` was oversampled slightly toward its natural occurrence rate in the wild. No class is below 20% or above 70%, keeping the classifier from defaulting to a majority class.

### Labeling process

Each example was assigned a label by applying the taxonomy from `planning.md` §2 and the edge case rules from §3 to every post. The first 50 examples were AI pre-labeled with the taxonomy in the LLM prompt; every suggested label was reviewed and confirmed before being recorded. The remaining 170 examples were labeled directly without AI assistance. The `suggested_label` column in `data/labeled_data.csv` identifies AI-pre-labeled rows; `human_overrode` tracks disagreements.

### Difficult examples

Three posts that genuinely required the edge case decision rules to classify:

| # | Post text (truncated) | Could be | Decision | Rule applied |
|---|---|---|---|---|
| 1 | "Karl Malone deserved the MVP award over Jordan in 1998... Malone had a better regular season by every statistical measure." | `analysis` or `hot_take` | `hot_take` | "Better by every statistical measure" is asserted, not demonstrated. No specific stats are cited; no peer comparison is made. Removing the evidence framing leaves only an opinion with nothing standing behind it. |
| 2 | "The refs decided that series... Every call went one way and anyone with eyes could see it." | `reaction` or `hot_take` | `hot_take` | References a specific series but the claim "every call went one way" characterizes an entire series, not a single play. Survives removal of the series reference — could be written about any series. |
| 3 | "The play-in loss hurts worse than a first-round exit... Getting knocked out before the bracket even starts feels uniquely embarrassing." | `reaction` or `hot_take` | `hot_take` | Likely prompted by a specific loss, but the claim is a general opinion about playoff formats that generalizes entirely independently of the triggering event. |

---

## Model

- **Base model:** `distilbert-base-uncased` (66M parameters)
- **Fine-tuning platform:** Google Colab (T4 GPU)
- **Training approach:** Linear classification head (hidden size → 3 logits) added to the pooled output; fine-tuned end-to-end with cross-entropy loss. Label map: `analysis=0`, `hot_take=1`, `reaction=2`.

| Hyperparameter | Value | Rationale |
|---|---|---|
| Epochs | 3 | Default in the starter notebook; loss converged by epoch 2, third epoch added minimal improvement |
| Learning rate | 2e-5 | Standard for DistilBERT fine-tuning; lower rates underfit on 154 examples |
| Batch size | 16 | Fits T4 VRAM comfortably with 512-token max length |
| Max sequence length | 512 | DistilBERT's context window; all 220 examples fit within this limit |

---

## Baseline

Zero-shot classification with Groq `llama-3.3-70b-versatile`. The prompt describes all three labels and both edge case rules and requests a structured `Label:` / `Reasoning:` response. No labeled examples were provided — the model relies entirely on its understanding of the taxonomy description.

**Baseline system prompt:**
```
You are classifying /r/nba posts and comments into one of three labels.

analysis: A structured argument using specific, verifiable evidence — a stat,
historical comparison, or tactical breakdown — where the evidence does genuine
argumentative work.

hot_take: A bold, confident claim stated without supporting evidence, or with
decorative evidence that does not genuinely support the argument. Asserts rather
than reasons.

reaction: An immediate emotional response to a specific recent event. Little to
no argument; the post would not exist without that specific event.

Respond with exactly:
Label: <analysis|hot_take|reaction>
Reasoning: <one sentence>
```

The baseline and the fine-tuned model ran on the identical held-out test set (33 examples). Neither model saw the test set during training or prompt design.

---

## Results

### Summary

| Metric | Fine-tuned DistilBERT | Groq Zero-Shot | Improvement |
|---|---|---|---|
| Overall accuracy | **84.8%** (28/33) | 66.7% (22/33) | +18.2 pp |
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

Rows = true label, columns = predicted label. Diagonal = correct predictions.

| True \ Predicted | `analysis` | `hot_take` | `reaction` |
|---|---|---|---|
| **`analysis`** | **8** | 2 | 0 |
| **`hot_take`** | 1 | **11** | 1 |
| **`reaction`** | 0 | 1 | **9** |

See `outputs/confusion_matrix.png` for a side-by-side visual comparison of both models.

### Success criteria check

All five numeric thresholds defined in `planning.md` §6 were met before seeing the test set.

| Criterion | Threshold | Result | Pass? |
|---|---|---|---|
| Overall accuracy | ≥ 70% | 84.8% | ✅ |
| Macro-averaged F1 | ≥ 0.65 | 0.852 | ✅ |
| `analysis` F1 | ≥ 0.60 | 0.842 | ✅ |
| `hot_take` F1 | ≥ 0.65 | 0.815 | ✅ |
| `reaction` F1 | ≥ 0.65 | 0.900 | ✅ |
| Fine-tuned macro-F1 beats baseline by ≥ 5 points | ≥ +0.05 | +0.186 | ✅ |

---

## AI-Assisted Pattern Analysis

Before writing the failure analysis below, I pasted all 5 wrong predictions into Claude and asked it to identify common themes — similar framing, vocabulary patterns, length, or a specific label pair that kept getting confused. Here is what it found, and what I verified or discarded.

**What the AI identified:**

1. *Analysis posts without explicit numeric statistics are mislabeled as hot\_take.* Two of the five errors (rnba_019, rnba_031) are analysis posts that reference concepts like "shot-selection models" and "motion principles" but lean on relational language rather than specific numbers. The AI suggested this was the single most common error pattern.

2. *Hot\_take posts with adverbial magnitude markers are mislabeled as analysis.* rnba_127 ("Historically bad when his team needed him most") uses "historically" as a rhetorical intensifier rather than a historical data claim. The AI flagged that "historically," "statistically," and similar adverbs may have been strong correlates for `analysis` in training data.

3. *Hot\_take posts with interrogative/exclamatory register are mislabeled as reaction.* rnba_113 ("How many times does the same big market team end up with the top pick when they need it most? You cannot tell me there is no thumb on the scale.") has the emotional punctuation pattern of a reaction post even though it argues a general claim about the league.

4. *Reaction posts with career-level evaluations are mislabeled as hot\_take.* rnba_197 uses "the best I have seen from him" — a claim spanning the player's career, not just tonight's game. The AI correctly identified this as the hardest case: the generalizing language is real, making the reaction classification legitimately ambiguous.

**What I verified:** Patterns 1 and 2 are accurate and reflect the same root cause (model learned vocabulary features of analytical writing rather than argumentative structure). Pattern 4 is accurate and matches the annotation note in planning.md. I confirmed these by re-reading the posts independently before writing the analysis below.

**What I discarded:** Pattern 3 is partially correct but overstated. rnba_113 is a real classification error, but interrogative/exclamatory register is not the primary cause — the post is making a general claim about league-wide patterns, which is a hot\_take by definition regardless of punctuation. The deeper issue is that the model has not learned to check whether an exclamatory post is making a general or event-specific claim.

---

## Failure Analysis

5 of 33 test examples were misclassified. Three are analyzed in depth.

### Error 1 — `analysis` predicted as `hot_take` (rnba_019)

**Post text:**
> "Kyrie's shot chart shows an extremely high volume of contested mid-range and pull-up attempts that most shot-selection models mark as inefficient. His actual efficiency on those shots is in the 95th percentile for the positions he takes them from — which is either extraordinary skill or a massive sample-size anomaly."

**True label:** `analysis` — **Predicted:** `hot_take`

**Which labels are being confused?** The most common error direction overall: `analysis` → `hot_take`. Both analysis misclassifications (rnba_019, rnba_031) went this direction. The model is being too strict about what counts as analysis — it is treating `hot_take` as the default for posts that make a claim without asserting a definitive conclusion.

**Why is that boundary hard?** This post references shot-selection models and 95th-percentile efficiency, but the framing is explicitly uncertain: "which is either extraordinary skill or a massive sample-size anomaly." It presents evidence and declines to conclude — which is actually a more rigorous form of analysis than the typical training example. The model appears to have learned that analysis posts assert a conclusion supported by evidence. When a post offers evidence and hedges, it reads as an opinion with a caveat rather than a structured argument.

**Is this a labeling problem or a data problem?** Data distribution problem. The annotation is consistent — this is clearly analysis by the taxonomy definition (the shot-selection model reference is verifiable and does genuine argumentative work). The issue is that most analysis examples in training have the structure "here is a stat → here is a conclusion the stat supports." This post has the structure "here is an observation → here are two possible explanations." The model has not seen enough training examples of evidence-without-conclusion analysis to learn that pattern.

**What would fix it?** More analysis training examples that present evidence without committing to a conclusion — posts that reason by elimination or lay out multiple interpretations of data. Currently those cases are underrepresented relative to the "evidence supports X" template.

---

### Error 2 — `hot_take` predicted as `analysis` (rnba_127)

**Post text:**
> "Playoff basketball is when we find out who the real players are, and Harden has failed that test every single time it counted. Great numbers during the regular season. Historically bad when his team needed him most."

**True label:** `hot_take` — **Predicted:** `analysis`

**Which labels are being confused?** The reverse direction: `hot_take` → `analysis`. This is the rarest error type (1 of 5), but it reveals an important boundary problem. The post is a textbook hot\_take — bold claim, no evidence, asserts rather than reasons — yet it got labeled as analysis.

**Why is that boundary hard?** The word "historically" carries strong analytical connotations even when used rhetorically. "Historically bad" does not cite any historical data; it is a rhetorical intensifier meaning "reliably" or "famously." But the model has likely learned that "historically" in training data correlates with analytical framing — posts that reference historical championship trends, historical scoring records, or historical franchise data. This is a false positive driven by a single word acting as a strong feature.

**Is this a labeling problem or a data problem?** Data distribution problem. The taxonomy is clear: this is a hot\_take (bold claim, no evidence, general assertion). But the training data may not include enough hot\_take examples that use academic-sounding vocabulary. Most hot\_take examples in this dataset use explicitly opinionated language ("is overrated," "will never," "is a bust"). If the model rarely sees confident claims phrased with hedging or formal vocabulary, it misroutes them.

**What would fix it?** More hot\_take training examples that use analytical-sounding language rhetorically. Specific targets: posts that say "historically" or "statistically" without citing actual data, posts that use "data shows" without providing any data, posts that frame opinions as facts using elevated vocabulary. These are the hardest hot\_takes to classify and the model has not learned them.

---

### Error 3 — `reaction` predicted as `hot_take` (rnba_197)

**Post text:**
> "Whatever happens in this series, tonight's performance from Tyrese Haliburton is the best I have seen from him. Ice in his veins in the final four minutes. He looked like a different player."

**True label:** `reaction` — **Predicted:** `hot_take`

**Which labels are being confused?** `reaction` → `hot_take`. One of two errors in this direction (rnba_197 and rnba_113). The model overpredicts hot\_take for reaction posts that contain generalizing evaluative language.

**Why is that boundary hard?** Two phrases generalize beyond tonight's game: "the best I have seen from him" spans the player's career, and "he looked like a different player" implies a persistent character change rather than a one-game observation. By the decision rule ("does the claim survive if you remove the game reference?"), those phrases survive — you could write "Haliburton has looked like a different player lately" without citing any specific game. The model correctly identified generalizing language and routed to hot\_take. The reason this is still labeled reaction is that the primary purpose of the post — the reason it was written — is to respond to a specific recent game. The general claims are incidental to the event-triggered purpose.

**Is this a labeling problem or a data problem?** Genuine boundary case. This post was documented as a difficult annotation case during labeling (see planning.md §3). A reasonable annotator could label it hot\_take and not be wrong — the taxonomy does not fully specify what to do when a post's primary purpose is event-reaction but its language generalizes. The annotation note says: "the generalizing language is incidental to the event-triggered purpose" — but this is a judgment call, not a deterministic rule.

**What would fix it?** Tightening the taxonomy: add an explicit rule for posts where event-triggered content is primary but includes one or two generalizing claims. Current options: label by the dominant purpose (reaction) or by whether any claim survives the removal test (hot\_take). Documenting this as a deliberate choice in planning.md rather than leaving it ambiguous would produce more consistent training signal.

---

## Sample Classifications

Five posts classified by the fine-tuned model with confidence scores (softmax probabilities across all three classes).

| Post (truncated) | True | Predicted | `analysis` | `hot_take` | `reaction` |
|---|---|---|---|---|---|
| "CURRY FROM HALF COURT AT THE BUZZER. NO WAY. I AM LOSING MY MIND..." | reaction | **reaction** ✅ | 0.01 | 0.02 | **0.97** |
| "Jokic's true shooting percentage this season is 66.8% on enormous volume. Most All-Star-caliber centers with similar usage land in the 58-62% range..." | analysis | **analysis** ✅ | **0.91** | 0.08 | 0.01 |
| "Kobe Bryant was a better player than LeBron James. Better scorer, better clutch player, better competitor. The stats favor LeBron but stats cannot measure heart..." | hot_take | **hot_take** ✅ | 0.06 | **0.89** | 0.05 |
| "Kyrie's shot chart shows an extremely high volume of contested mid-range attempts that most shot-selection models mark as inefficient. His actual efficiency is in the 95th percentile..." | analysis | **hot_take** ❌ | 0.29 | **0.67** | 0.04 |
| "The series is tied 3-3. These are my favorite words in sports. Six games of basketball with everything on the line in one winner-take-all game. I live for this." | reaction | **reaction** ✅ | 0.08 | 0.18 | **0.74** |

**Why Example 2 is a reasonable correct prediction:** Jokic's true shooting percentage (66.8%) is a specific, verifiable statistic; the comparison group (All-Star-caliber centers with similar usage, 58–62% range) is precisely defined; and the causal explanation (FTr of 0.41 compounds the efficiency advantage) does genuine argumentative work. The model's high confidence (0.91) reflects that this post hits all the features it learned to associate with analysis: a specific number, a comparison, and a conclusion the evidence supports.

**What Example 4 reveals:** The model's 0.67 hot\_take confidence on rnba_019 is notably lower than its confidence on clear cases (0.89–0.97). This uncertainty is accurate — the post is at a genuine boundary. The model is not confidently wrong; it is uncertain in the right place. A confidence threshold classifier (e.g., flag anything below 0.75 for human review) would correctly identify this as a borderline case.

**Example 5 note:** The lower reaction confidence (0.74) on the "3-3 series" post reflects that "these are my favorite words in sports" is a generalizing personal preference rather than event-specific content. The model distributes some probability to hot\_take (0.18), which is the correct response to ambiguous language.

---

## Reflection

**What the model captured vs. what I intended it to capture:**

The taxonomy was designed around argumentative structure — does the evidence do genuine work? The model learned vocabulary and surface form instead. That gap is the most important thing the evaluation revealed.

**`reaction` was learned correctly.** F1 of 0.900 and very high confidence on clear examples suggest the model found strong, stable features: ALL CAPS, exclamations, present-tense verbs ("just," "right now," "he goes"), event-specific proper nouns in an emotional register. These are reliable surface signals that align almost perfectly with the intended definition. What the model intended to learn and what it actually learned are the same here.

**`hot_take` was learned mostly correctly, with one failure mode.** The model correctly classifies hot\_takes that use opinionated language ("is overrated," "will never," "is a bust") and bold superlatives. Where it fails is hot\_takes that use analytical-sounding vocabulary rhetorically — "historically," "statistically," "data shows" — without citing actual data. These posts sound like analysis on the surface even when they assert without reasoning. The model overfit to the surface features of analytical writing and missed the substance test.

**`analysis` was learned partially.** The model learned to look for explicit statistics and comparative structures. Where it fails is analysis that does not follow the "stat → conclusion" template: posts that present evidence and reason about it without committing to a conclusion, or posts that explain a mechanism without a numeric anchor. These are valid analysis by the taxonomy definition, but the training data underrepresents them relative to the clean "cite a number, draw an inference" pattern.

**The root cause:** The training data was too consistent within each class. The analysis examples almost all contain explicit statistics; the hot\_take examples almost all lack them. This made the model's task easy (learn the presence or absence of numbers as a proxy for the label) without teaching the harder distinction the taxonomy actually encodes (does the evidence do argumentative work?). A dataset with more cross-class contamination — analysis posts without explicit statistics, hot\_takes that contain real data — would force the model to learn the actual signal.

**What fine-tuning added over zero-shot:** The baseline's errors were distributed across all three label pairs; it confused analysis and hot\_take, hot\_take and reaction, and analysis and reaction. The fine-tuned model's errors are almost entirely at the analysis/hot\_take boundary. This is progress: the model has learned that reaction is distinct and is now focused on the hardest boundary in the taxonomy. That concentration of errors is exactly what you would expect from a model that has learned the easier distinctions and is now failing only at the hard ones.

---

## Spec Reflection

**How planning.md guided implementation:**

The edge case rules in planning.md §3 were the most directly useful part of the spec. Having written-down decision rules ("would a statistician call this evidence or a rhetorical flourish?") meant that annotation decisions for borderline posts were deterministic rather than improvised. When rnba_184 ("The play-in loss hurts worse than a first-round exit") came up during annotation, the rule "does the claim survive if you remove the game reference?" gave an immediate answer: yes → hot\_take. Without the written rule, this would have required re-reading the full taxonomy each time. The spec converted a judgment call into an application of a specific test.

The success criteria in planning.md §6 were equally important in a different way. Having specific numeric thresholds written before training prevented post-hoc rationalization of the results. When the fine-tuned model came back with analysis F1 of 0.842, there was a clear, pre-committed answer to whether that was good enough (yes — threshold was 0.60). Without pre-committed thresholds, it would be easy to argue that 0.842 was either a success or a failure depending on what you were hoping for.

**Where implementation diverged from the spec:**

The AI Tool Plan in planning.md §7 specified running a label stress-test before annotation: generate 10–15 boundary posts, apply the edge case rules, and tighten definitions if more than 2 of 10 are unclassifiable. This step was skipped in favor of going directly to data collection.

In retrospect, the stress-test would have caught the specific problem that showed up in the evaluation: analysis posts without explicit statistics are ambiguous under the current definition. A stress-test prompt asking "generate 10 posts that sit at the exact boundary between hot\_take and analysis" would almost certainly have produced posts like rnba_019 — evidence presented with uncertainty rather than as support for a conclusion. If those posts proved hard to classify, the definition could have been tightened before 154 training examples were annotated with the ambiguous rule. The cost of that shortcut showed up as 2 of 5 test errors.

---

## AI Usage

**Instance 1 — Dataset generation**

*What I directed:* I asked Claude Sonnet 4.6 to generate 220 r/nba-style posts and comments across three label categories, using the label definitions and edge case rules from planning.md §2–3 as the generation prompt. I specified target counts (65 analysis, 90 hot\_take, 65 reaction) and asked for variety in topic, length, and community vernacular (including ALL CAPS emotional reactions, statistical analysis posts, and confident opinionated claims).

*What it produced:* 220 posts matching the requested distribution, covering a broad range of NBA topics (player legacy debates, statistical breakdowns, game reactions, trade news, historical comparisons) with realistic sub-Reddit voice.

*What I changed or overrode:* The first 50 rows were reviewed for label accuracy before being recorded in the CSV. Two borderline cases were reconsidered and their annotations adjusted to be consistent with the edge case rules. One example referencing a fictional player name ("DeSean Washington-Ball Jr") was replaced with a more generic formulation to keep the data consistent with actual r/nba content conventions.

**Instance 2 — Failure pattern analysis**

*What I directed:* After running evaluation, I pasted all 5 wrong predictions (post text, true label, predicted label) into Claude and asked: "Look at these misclassified posts. What patterns do you see? Are certain types of language being systematically mislabeled? What does this suggest about what the model learned?"

*What it produced:* Four candidate patterns (analysis without explicit statistics → hot\_take; adverbial magnitude markers in hot\_takes → analysis; interrogative/exclamatory hot\_takes → reaction; reaction posts with career-level evaluations → hot\_take). It also suggested the root cause was "vocabulary correlation rather than argumentative structure."

*What I changed or overrode:* Pattern 3 (interrogative register → reaction confusion) was discarded as the primary explanation for rnba_113. The actual problem is that the model has not learned to check whether an exclamatory post makes a general or event-specific claim — not that punctuation patterns directly predict reaction. I accepted patterns 1, 2, and 4 as accurate after re-reading the posts independently, and used them to structure the failure analysis above. The "vocabulary correlation" framing from the AI was accurate and became the organizing concept for the Reflection section, but the specific language in the Reflection was written after verifying the pattern directly rather than copied from the AI output.

**Instance 3 — Pre-labeling (first 50 examples)**

*What I directed:* For the first 50 examples in labeled\_data.csv, I sent each post to Groq `llama-3.3-70b-versatile` with the three label definitions and edge case rules in the system message and requested `Label: <label>\nReasoning: <one sentence>` format.

*What it produced:* Suggested labels for all 50 examples, stored in the `suggested_label` column.

*What I changed or overrode:* Every suggested label was reviewed before being recorded in the final `label` column. Two labels were overridden where the LLM applied the wrong edge case rule (one post the LLM labeled `reaction` was relabeled `hot_take` because its claim survived removal of the event reference; one post labeled `hot_take` was relabeled `analysis` because the stat it cited was used as genuine evidence for a testable conclusion). Override rate: 4% (2/50), consistent with the LLM understanding the taxonomy but occasionally missing the harder boundary.

---

## Demo Video

The demo video should be recorded in Google Colab by running inference on 3–5 held-out posts and printing the predicted label and confidence score for each. Key moments to capture:

1. Run at least 3 posts through the fine-tuned model with label and confidence visible in the output
2. Narrate one correct prediction — explain which features in the post match the predicted label
3. Narrate one incorrect prediction (rnba_019 is recommended) — explain why the post is correctly labeled `analysis` but was predicted as `hot_take`
4. Walk through the confusion matrix and summary table in this README

---

## Repo Structure

```
ai201-project3-takemeter/
├── data/
│   └── labeled_data.csv         # 220 annotated examples with suggested_label and human_overrode columns
├── outputs/
│   ├── evaluation_results.json  # per-example predictions and aggregate metrics for both models
│   └── confusion_matrix.png     # side-by-side confusion matrix visual (fine-tuned vs. baseline)
├── planning.md                  # design doc: taxonomy, edge cases, data plan, metrics, success criteria, AI tool plan
├── COMPANION.md                 # pipeline build log: step-by-step progress, decisions, known risks
└── README.md                    # this file — final project report
```
