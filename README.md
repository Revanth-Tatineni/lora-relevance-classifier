# lora-relevance-classifier

[![CI](https://github.com/Revanth-Tatineni/lora-relevance-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Revanth-Tatineni/lora-relevance-classifier/actions/workflows/ci.yml)

LoRA-fine-tunes a small open model to answer one question: given a question and a candidate
chunk of text, does that chunk actually answer it? This is exactly the decision
[rag-qa-api](https://github.com/Revanth-Tatineni/rag-qa-api)'s relevance gate makes with a
fixed `MIN_RERANK_SCORE` threshold today — this project turns it into a supervised
classification task and fine-tunes a learned alternative.

```bash
python -m lora_relevance_classifier.build_dataset --source /path/to/rag-qa-api/evals/dataset.jsonl
lora-relevance-classifier train --base-model Qwen/Qwen2.5-0.5B --epochs 5
lora-relevance-classifier compare --adapter-dir runs/lora-adapter/adapter
```

## Why this exists, and why last

Project 5 of 5. It's the most standalone project on the list — it doesn't need projects 1–4
running to function — but it's deliberately not the task a disconnected fine-tuning tutorial
would pick. The training data comes directly from `rag-qa-api`'s own answerability eval
dataset (22 answerable + 5 unanswerable IRS-publication questions, with real evidence quotes),
turned into question/context pairs. The result is meant to plug back into that project's
relevance gate, not stand alone as a toy classifier.

## Why this isn't a trivial fine-tune

- **Real task framing.** "Is this question answerable from this chunk" is reframed as sentence-
  pair classification — the exact shape of decision a cross-encoder reranker already makes in
  `rag-qa-api`, which is why a fine-tuned model is a plausible drop-in, not a mismatched idea
  bolted on afterward.
- **Hard negatives, not just random ones.** Each answerable question is paired not only with
  its own evidence (positive) but with *other* questions' real evidence (hard negative) — text
  that's topically similar (all from IRS publications) but doesn't answer this specific
  question. That's the case a weak classifier gets wrong; random unrelated text is too easy.
- **A measured baseline, not just a claimed improvement.** `baseline.py` is a zero-training
  word-overlap heuristic — the "before" a fine-tune has to beat. See the honest caveat about
  what it's a proxy *for* in `baseline.py`'s own docstring and `docs/explained/`.
- **False-positive rate is reported, not just accuracy.** In this task, a false positive (the
  model says "relevant" when it isn't) is the dangerous direction — it's what lets an
  unsupported answer through `rag-qa-api`'s relevance gate. `metrics.py` reports it explicitly.

## Stack

PyTorch (CPU for the pipeline/tests; the real training run needs a GPU), HuggingFace
`transformers` + `peft` (LoRA) + `datasets`, hand-rolled classification metrics (four numbers
don't need scikit-learn as a dependency), pytest, ruff, mypy, GitHub Actions. The real target
model is `Qwen/Qwen2.5-0.5B`, fine-tuned on a free-tier GPU (Colab or Kaggle) — see
`notebooks/train_on_colab.md`.

## Status

- **Code complete and genuinely tested**, including one test that runs a real, full LoRA
  fine-tune — `peft.get_peft_model`, a real `transformers.Trainer.train()` loop, a saved and
  reloaded adapter — end to end on CPU, using a tiny proxy model
  (`hf-internal-testing/tiny-random-gpt2`), not the real target model. This proves the pipeline
  is mechanically correct, not that the real model has been fine-tuned.
- **The baseline row below is real**, run against the actual data: building the dataset from
  `rag-qa-api`'s real `evals/dataset.jsonl` (27 source questions) produced 77 question/context
  pairs (61 train, 16 val), and the word-overlap baseline on the real validation split scored:

  | | Accuracy | Precision | Recall | F1 | False-positive rate |
  |---|---|---|---|---|---|
  | Word-overlap baseline (real) | 0.625 | 1.0 | 0.143 | 0.25 | 0.0 |
  | LoRA-fine-tuned (`Qwen/Qwen2.5-0.5B`) | RESULTS_PLACEHOLDER | RESULTS_PLACEHOLDER | RESULTS_PLACEHOLDER | RESULTS_PLACEHOLDER | RESULTS_PLACEHOLDER |

  Reading the baseline honestly: it never produces a false positive (precision 1.0, FPR 0.0)
  but misses most real matches (recall 0.143) — IRS questions and their actual evidence often
  share almost no literal words (e.g. "standard deduction for a single filer" vs. the evidence
  "Single or Married filing separately $15,750"). That gap between high precision and low
  recall is exactly the opening a fine-tuned model would need to close to be worth using.
- **There is deliberately no "base model, zero-shot" row.** `AutoModelForSequenceClassification`
  on an un-fine-tuned base model has a randomly-initialized classification head — its
  predictions would just be noise, not a meaningful zero-shot judgment. A real zero-shot
  comparison would need a *generative* prompt ("does this text answer this question?") parsed
  for yes/no, which is a different code path than the classification head LoRA trains here and
  hasn't been built. Comparing against a random head and calling it "zero-shot" would be
  misleading, so it's left out rather than faked.
- **Not yet done, honestly:** the real target model (`Qwen/Qwen2.5-0.5B`) has never been
  fine-tuned. That needs a GPU (Colab/Kaggle free tier) and hasn't been run yet — the
  LoRA-fine-tuned row above stays a placeholder until it is.
- **Known, accepted limitation:** 27 source questions is a small seed dataset. The
  pairing/hard-negative strategy in `dataset.py` multiplies that into 77 pairs, but this is
  still a small-data fine-tune, not a production-scale one — the honest claim from this project
  is "I built and ran a correct, tested LoRA fine-tuning pipeline," not "I shipped a
  production-grade classifier from 27 questions."
- **Also not yet done:** comparing against `rag-qa-api`'s *real* cross-encoder rerank scores
  (the actual thing this project's classifier would replace) needs that project's retrieval
  stack running to produce them. `baseline.py`'s word-overlap heuristic is a stand-in until
  that integration exists.
- No git commit, no push, no deployment yet.
