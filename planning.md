# TakeMeter — Planning Document
**AI201 Project 3 · Fine-Tuned Text Classifier · /r/nba**

---

## 1. Community

**Chosen community:** r/nba

r/nba is one of the largest and most active sports communities on Reddit, with several million members producing thousands of posts and comments daily. It was chosen for three reasons.

First, the discourse is genuinely varied. A single thread about a game will contain a mix of emotional exclamations, statistical arguments, and sweeping claims with no supporting evidence — all in the same place, often from the same person in adjacent comments. That range is rare and makes the community unusually rich for a classification task.

Second, the distinctions being measured matter to participants. r/nba regulars actively police discourse quality: "that's just a hot take," "cite your sources," and "this is such a reaction post" are common community responses. The labels aren't being imposed from outside — they map to distinctions the community already makes.

Third, the text is short, opinionated, and linguistically consistent enough for a small classifier to learn from. NBA posts aren't long-form essays — they're punchy, direct, and usually make a single claim. That makes the classification signal cleaner than, say, a policy or academic community where arguments are multi-part and hedged.

---

## 2. Label Taxonomy

Three labels. The central axis: **how substantiated is the claim?**

**Design constraints:**
- Mutually exclusive: every post belongs to exactly one label
- Exhaustive: covers ≥90% of text posts and comments without a catch-all bucket
- Grounded in r/nba norms: the distinctions must match how participants already talk about post quality

---

### `analysis`

A post makes a structured argument for a claim using at least one piece of specific, verifiable evidence — a statistic, historical comparison, or tactical breakdown — where the evidence does genuine argumentative work. A reader could check the data, and removing the evidence would weaken the argument.

**Example 1:**
> "People act like defensive teams always win titles, but since 2015, five of ten championships went to top-five offenses versus only three to top-five defenses. The '3-point era' narrative about defense is just wrong — shot quality and pace matter more now."

**Example 2:**
> "Jokic's on/off differential this season is +9.4, meaning his teammates are a league-average unit without him. Compare that to the next best big man at +5.1. The MVP conversation shouldn't even be close."

---

### `hot_take`

A bold, confident claim about a player, team, or basketball topic stated without supporting evidence — or with decorative evidence that doesn't genuinely support the argument. The post asserts rather than reasons. Claims are general (not tied to a single event) and could be written without any recent game having happened.

**Example 1:**
> "LeBron is clearly past his prime and has been for three years. His playoff runs just don't carry teams anymore. Anyone who thinks he's still top-3 in the league is coping."

**Example 2:**
> "The three-point revolution ruined basketball. Centers used to have real post games. The game was more beautiful to watch before Steph happened."

---

### `reaction`

An immediate emotional response to a specific, recent event — a game, a play, a trade, or breaking news. Little to no argument. The post expresses a feeling about something that just happened and would not exist without that specific event. Often time-sensitive; context-dependent.

**Example 1:**
> "CURRY WITH THE 3 AT THE BUZZER NO WAY. THIS MAN IS NOT REAL."

**Example 2:**
> "That flagrant foul call in Q4 was the worst officiating I've seen this season. Refs literally decided that series. Disgusting."

---

## 3. Hard Edge Cases

### Primary boundary: `hot_take` vs `analysis`

**The case:** A post makes a bold claim and includes one statistic. Is the stat doing genuine argumentative work, or is it decorative?

> "LeBron is overrated — his playoff win rate against top-seeded opponents is below .500."

This feels like analysis because it cites a number. But the stat is cherry-picked, presented with no context (sample size? which era? how does it compare to peers?), and the overall argument is accusatory framing first, evidence second.

**Decision rule:** If removing the opinion framing would leave a verifiable, coherent argument standing, label it `analysis`. If the stat is present to sound credible rather than to genuinely reason — one number, no context, no alternative explanations — label it `hot_take`. The test is: *would a statistician call this evidence, or a rhetorical flourish?*

---

### Secondary boundary: `reaction` vs `hot_take`

**The case:** A game happens, and someone writes a sweeping claim about a player's character prompted by that game.

> "After that game 7 choke, I'm convinced [player] just doesn't have a killer mentality."

This was triggered by a specific event (the game), but the claim is general (a statement about the player's character across their career).

**Decision rule:** If the claim could be written independently of the specific event that prompted it — if it's a general statement about a player or team that happens to reference a game — label it `hot_take`. If the post is purely about the specific event and doesn't generalize beyond it, label it `reaction`. The test is: *does the claim survive if you remove the game reference?*

---

### Annotation note

When a post lands on a hard boundary, write the example down in the Difficult Examples section of this document with the rule you applied. After 10 difficult cases, review whether the rules need updating before continuing.

---

## 4. Data Collection Plan

**Sources:**
- r/nba top posts of the week — text posts primarily (`analysis`, `hot_take`)
- r/nba game thread comments — `reaction` and `hot_take`
- r/nba post-game discussion threads — all three labels
- r/nba "Hot Takes" or "Unpopular Opinion" megathreads (posted periodically) — `hot_take`
- r/nba "Film Room" or player breakdown posts — `analysis`

**Scope:** Text posts and comments only. Link posts, image posts, highlight posts, and meme posts are excluded.

**Targets per label:**

| Label | Target count | % of 200 |
|---|---|---|
| `analysis` | 60 | 30% |
| `hot_take` | 80 | 40% |
| `reaction` | 60 | 30% |
| **Total** | **200** | **100%** |

`hot_take` is the most naturally occurring label in the wild. `analysis` and `reaction` require more deliberate sourcing.

**If a label is underrepresented after 200 examples:**

If `analysis` is below 40 examples: actively seek Film Room posts, statistical breakdown threads, and "prove me wrong with data" style posts. Do not lower the annotation standard — a post with only one decorative stat is still `hot_take`.

If `reaction` is below 40 examples: go to game thread comments directly; reactions are almost exclusively found there, not in top-level posts.

If `hot_take` is below 40 examples (unlikely): pull from "unpopular opinion" threads and player comparison posts.

**Split:** 70% train (~140) / 15% validation (~30) / 15% test (~30), stratified by label.

---

## 5. Evaluation Metrics

### Why accuracy alone is not enough

The three labels will not be perfectly balanced in the test set. If the model learns to predict `hot_take` for everything, it could achieve ~40% accuracy while completely failing to identify `analysis` or `reaction`. Accuracy would mislead you into thinking the model has learned something useful.

More importantly, the costs of different errors are not equal. Misclassifying a `reaction` as a `hot_take` is a minor error — both lack evidence. Misclassifying an `analysis` post as a `hot_take` is a more meaningful failure — it means the model can't recognize the thing the classifier is supposed to reward.

### Metrics used

**Per-class F1 (primary metric):**
F1 = harmonic mean of precision and recall for each label. Used as the primary metric because it penalizes a model that gets high precision by only predicting a label when very confident (missing true positives) and also penalizes high recall achieved by predicting a label too often (false positives). Per-class F1 exposes whether the model has actually learned all three categories or is collapsing them.

**Macro-averaged F1 (summary metric):**
Averages F1 across all three classes without weighting by class size. This treats `analysis` performance as equally important as `hot_take` performance, which is the right framing — a useful classifier must work on all three labels, not just the majority one.

**Overall accuracy (context metric):**
Reported alongside macro-F1 for context, but not used as the primary success criterion.

**Confusion matrix:**
Shows which specific misclassifications are most common. Especially useful for confirming whether the `analysis` vs `hot_take` boundary is being learned correctly — the most common error is expected to be `hot_take` posts being labeled `analysis` (false promotion).

---

## 6. Definition of Success

A classifier is useful in a real community tool if it can reliably distinguish evidence-backed posts from unsupported assertions — giving moderators, readers, or recommendation systems a signal about post quality.

**Minimum thresholds for "good enough to deploy":**

| Metric | Threshold |
|---|---|
| Overall accuracy | ≥ 70% on held-out test set |
| Macro-averaged F1 | ≥ 0.65 |
| `analysis` F1 | ≥ 0.60 (the hardest and most important class) |
| `hot_take` F1 | ≥ 0.65 |
| `reaction` F1 | ≥ 0.65 |
| Fine-tuned model vs. baseline | Fine-tuned must outperform Groq zero-shot on macro-F1 by ≥ 5 points |

**Rationale for these specific numbers:**

70% overall accuracy is above the majority-class baseline (~40% if everything were `hot_take`). It's the floor at which the classifier is doing something non-trivial.

`analysis` F1 ≥ 0.60 is set lower than the other two because `analysis` is the hardest class to learn: its boundary with `hot_take` is subtle and it may be underrepresented in training data. 0.60 means the model is getting the majority of `analysis` posts right even if it misses some borderline cases.

The +5 macro-F1 gap over the Groq baseline is the real success criterion. Fine-tuning should outperform zero-shot on a domain-specific task with labeled examples. If it doesn't, the labels are probably not precise enough for the model to learn from.

**Objective check:** At evaluation time, compute these five numbers from `evaluation_results.json`. Pass/fail on each threshold is a binary decision — no subjective interpretation required.

---

## 7. AI Tool Plan

### Label stress-testing (before annotation)

**What:** Give an AI tool the three label definitions and both edge case rules and ask it to generate 10 posts that sit at the boundary between `hot_take` and `analysis`, and 5 posts that sit at the boundary between `reaction` and `hot_take`.

**Why:** If the generated posts can't be cleanly classified using the current definitions, the definitions need tightening before 200 examples are annotated. Fixing definitions after annotation means re-labeling.

**How:** Paste the label definitions from Section 2 and the rules from Section 3 into a prompt. Ask: "Generate 10 r/nba posts that sit at the exact boundary between hot_take and analysis. Each post should make it genuinely unclear which label applies." Review each generated post against the decision rules. If more than 2 of 10 are unclassifiable, revise the rules.

**Decision:** Run this before starting annotation. Update Section 3 if the rules need sharpening.

---

### Annotation assistance

**What:** Use an LLM to pre-label up to 50 examples before reviewing them manually. The LLM's labels are treated as a first pass — each one is reviewed and either confirmed or overridden.

**Why:** Pre-labeling speeds up annotation for clear cases (unambiguous `reaction` or obvious `hot_take` posts), freeing attention for the genuinely hard ones.

**How:**
- Send each post to Groq `llama-3.3-70b-versatile` with the three label definitions and edge case rules in the system message
- Request `Label: <label>\nReasoning: <one sentence>` format
- Record the LLM's suggested label in a `suggested_label` column in the CSV
- Manually review every suggested label and record your final decision in `label`
- Track disagreements in a `human_overrode` boolean column

**Disclosure:** The CSV will include a `suggested_label` column for any pre-labeled examples. This will be noted in the README's data collection section.

**Decision:** Use pre-labeling for the first 50 examples as a calibration exercise, then annotate the remaining 150 manually. Compare agreement rates to see how well the LLM's classifications match the taxonomy.

---

### Failure analysis

**What:** After running evaluation, give the list of wrong predictions to an AI tool and ask it to identify patterns before writing up the analysis.

**Why:** Individual wrong predictions are easy to understand in isolation. Patterns across 15–20 errors are harder to see without help. An LLM can cluster errors by type and surface hypotheses worth investigating.

**How:**
- Extract all misclassified examples from `evaluation_results.json`
- Format as a list: post text, ground truth label, predicted label
- Ask: "Look at these misclassified posts. What patterns do you see? Are certain types of language being systematically mislabeled? What does this suggest about what the model learned?"
- Use the LLM's hypotheses as a starting point — then verify each pattern manually by reading the posts yourself before writing the evaluation report

**What to look for:**
- Are all `analysis` misclassifications going to `hot_take`, or are some going to `reaction`?
- Are the misclassified `analysis` posts the ones with a single decorative stat (the hard boundary case)?
- Are `reaction` posts being confused with `hot_take` when they contain a general claim prompted by an event?

**Decision:** Run this for every wrong prediction in the test set. Do not include the LLM's pattern descriptions verbatim — use them as hypotheses and verify by re-reading the posts.

---

## Difficult Examples

| # | Post text (truncated) | Could be | Decision | Rule applied |
|---|---|---|---|---|
| 1 | "Karl Malone deserved the MVP award over Jordan in 1998... but Malone had a better regular season by every statistical measure." | `analysis` or `hot_take` | `hot_take` | "Better by every statistical measure" is asserted, not demonstrated — no specific stats are cited and no peer comparison is made. Removing the evidence framing leaves only an opinion with no coherent argument standing. The stat claim is decorative, not genuinely argumentative. |
| 2 | "The refs decided that series... Every call went one way and anyone with eyes could see it." | `reaction` or `hot_take` | `hot_take` | Although it references a specific recent series (the event), the claim "every call went one way" is a generalizing statement about an entire series, not a reaction to a single play or moment. It survives removal of the series reference — the same sentence could be written about any series. The post characterizes patterns across multiple games. |
| 3 | "The play-in loss hurts worse than a first-round exit. At least in the first round you are in the playoffs. Getting knocked out before the bracket even starts feels uniquely embarrassing." | `reaction` or `hot_take` | `hot_take` | Even though this was almost certainly prompted by a specific play-in loss, the claim is a general opinion about playoff formats and how different elimination types feel. It does not describe the specific game, a specific play, or a specific moment. The claim generalizes completely independently of the triggering event. |

---

## Fine-Tuning Pipeline

- **Base model:** `distilbert-base-uncased`
- **Platform:** Google Colab (free T4 GPU)
- **Libraries:** `transformers`, `datasets`, `scikit-learn`
- **Training approach:** *(document after training — learning rate, batch size, epochs)*
- **Key hyperparameter decisions:** *(fill in after training with rationale)*

---

## Baseline Comparison

- **Baseline model:** Groq `llama-3.3-70b-versatile`, zero-shot
- **Method:** Classify each test example with the taxonomy in the system message and no labeled examples
- **Groq prompt:** *(paste after writing it)*

---

## Output Files

| File | Contents |
|---|---|
| `data/labeled_data.csv` | 200+ annotated examples with `id`, `source_type`, `text`, `suggested_label`, `label`, `human_overrode` |
| `outputs/evaluation_results.json` | Per-example predictions, ground truth, and aggregate metrics (downloaded from Colab) |
| `outputs/confusion_matrix.png` | 3×3 confusion matrix (downloaded from Colab) |
