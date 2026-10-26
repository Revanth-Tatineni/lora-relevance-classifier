# What this project is, and why it exists

## In one paragraph

`lora-relevance-classifier` fine-tunes a small open model with LoRA to answer one question: given a
question and a candidate chunk of retrieved text, does that chunk actually answer the
question? That's exactly the decision `rag-qa-api`'s relevance gate makes today with a fixed
`MIN_RERANK_SCORE` cross-encoder threshold. This project reframes it as a supervised
sentence-pair classification task, builds a training set directly from `rag-qa-api`'s own
answerability eval data, and fine-tunes a learned alternative to that fixed threshold.

## Where it sits in the bigger plan

**Project 5 of 5** (`D:\personal\job_applie\projects\open-source-project-plan.md`). It's the
most standalone project on the list -- it doesn't need projects 1-4 running to function -- but
that's a statement about *infrastructure* dependency, not about the task being disconnected
from the rest of the portfolio. The plan's own text for project 5 names this exact task: *"a
small classifier that routes questions (answerable from the documents or not) could replace or
back up the score threshold, trained on the eval data"* (from `rag-qa-api`'s own companion doc,
§15, written while building project 2). This project does that, not a generic fine-tuning
tutorial task picked in isolation.

It goes last on purpose, despite being the shortest to build: doing it last means fine-tuning
on a *real* downstream need an earlier project actually surfaced (the relevance gate's fixed
threshold), rather than picking a toy task just to prove `peft` works. It's also meant to be the
freshest skill right before interviews -- LoRA rank, quantization, catastrophic forgetting are
easier to discuss in depth having just done it.

## Why this task, specifically

`rag-qa-api`'s relevance gate is a single number (a cross-encoder score) compared against a
fixed threshold, calibrated once from 27 labelled questions
(`evals/results/retrieval.md`'s "Relevance threshold" section). A fixed threshold is brittle --
it doesn't adapt to a new document corpus, and it can't learn from being wrong. Turning the
same underlying decision into a supervised classification task is the natural next step, and
it's one that reuses data that already exists rather than requiring a new, disconnected
dataset.

## Who it's for

- **A recruiter or interviewer** skimming the README: a concrete real baseline number already
  run against real data, an honest explanation of what's still a placeholder, and a clear line
  back to project 2.
- **An interviewer probing fine-tuning depth:** why hard negatives instead of random ones, why
  there's deliberately no "zero-shot base model" comparison row, why W&B wasn't wired in as a
  flag. All of that is in this `docs/explained/` folder.
- **Future Revanth**, about to run the real GPU training: `notebooks/train_on_colab.md` is the
  exact sequence of commands, written before the run, not reconstructed after it.

## Honest current status (as of this write-up)

- **Code complete and genuinely tested.** 22 tests pass, including one that runs a real,
  complete LoRA fine-tune -- `peft.get_peft_model`, a real `transformers.Trainer.train()` loop,
  a saved-and-reloaded adapter -- end to end on CPU, using a tiny proxy model
  (`hf-internal-testing/tiny-random-gpt2`), not the real target model. `ruff check`/`format
  --check` and `mypy src` are all clean.
- **The dataset-building step has been run for real**, against `rag-qa-api`'s actual
  `evals/dataset.jsonl`: 27 source questions became 77 question/context pairs (61 train, 16
  val).
- **The word-overlap baseline has been run for real**, on that real validation split:
  accuracy 0.625, precision 1.0, recall 0.143, F1 0.25, false-positive rate 0.0. This is a real
  number from real data, not a placeholder.
- **Not yet done:** the real target model (`Qwen/Qwen2.5-0.5B`) has never actually been
  fine-tuned -- that needs a GPU (Colab/Kaggle free tier), and the README's results table still
  has `RESULTS_PLACEHOLDER` in that row. Don't claim a specific improvement number in an
  interview until that run has actually happened.
- **Deliberately not built:** a "zero-shot base model" comparison row. A freshly-initialized
  sequence-classification head on an un-fine-tuned base model is random noise, not a meaningful
  zero-shot baseline -- see `01-architecture.md` for the reasoning and what a real zero-shot
  comparison would actually require.
- No git commit, no push, no deployment yet.
