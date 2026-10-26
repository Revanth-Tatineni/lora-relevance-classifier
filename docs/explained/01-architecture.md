# Architecture decisions, and the alternatives that were rejected

## Sentence-pair classification, not question-only classification

**Chosen:** the model sees *both* the question and a candidate context chunk
(`tokenizer(question, context, ...)`, the standard two-sequence input a sequence-classification
head expects) and predicts whether the context answers the question.

**Rejected:** classifying the question alone ("is this question answerable in general").

**Why:** the real system this replaces -- `rag-qa-api`'s relevance gate -- never asks "is this
question answerable" in the abstract; it asks "does *this specific retrieved chunk* answer
it," after retrieval has already happened. A question-only classifier would be learning a
different, less useful task (closer to "does this look like an IRS question" than "is this
retrieval any good"), and it couldn't reuse the cross-encoder's actual job.

## Hard negatives from other questions' real evidence, not random text

**Chosen:** `dataset.py`'s `build_pairs` pairs each answerable question with `hard_negatives_per_question`
other questions' *real* evidence snippets, not random unrelated sentences.

**Why:** the actual failure mode a relevance gate has to avoid is a chunk that's topically
plausible -- from the same corpus, about the same general subject (IRS tax rules) -- but
doesn't answer *this* question. Pairing a question with something from an unrelated domain
(e.g. random Wikipedia text) would be trivially easy to reject on surface features alone, and a
classifier that only learned to reject that would tell you nothing about whether it can handle
the real gate's hard cases. Random negatives were specifically rejected as too easy, even
though they'd be simpler to generate.

## Splitting by `source_id`, not by individual pair

**Chosen:** `split_pairs` groups pairs by the originating question's id before splitting, so
every pair derived from one question (its positive and all its hard negatives) ends up
entirely in train or entirely in val.

**Why:** splitting at the pair level would let the exact same question's wording appear in both
train and val (once paired with its real evidence, once with a hard negative) -- the model
could partly memorize that specific question's phrasing during training and look better on
validation than it would on a genuinely new question. Splitting by source question is what
makes the validation score mean "how does this do on a question it never saw," not "how does
this do on a sentence pair it never saw from a question it did."

## No "zero-shot base model" comparison row

**Rejected, explicitly, after consideration:** running the un-fine-tuned base model through
`AutoModelForSequenceClassification` and reporting its accuracy as a "zero-shot" baseline.

**Why rejected:** `AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)`
on a causal LM that was never trained for classification attaches a **newly, randomly
initialized** classification head (see the "MISSING: score.weight" load report any such call
prints) -- this was directly observed during `tests/test_train_smoke.py`'s tiny-model loading.
Its predictions before any fine-tuning are statistically indistinguishable from a coin flip,
regardless of how capable the underlying base model actually is at language understanding.
Reporting that number as "the base model's zero-shot performance" would be misleading -- it
measures a random head, not the model's judgment.

**What a real zero-shot baseline would need instead:** a *generative* prompt --
"Question: {q}\nContext: {c}\nDoes the context answer the question? Answer yes or no." -- run
through the base model's text-generation path, with the output parsed for yes/no. That's a
different code path from the classification-head architecture LoRA trains here (it would need
an instruction-tuned variant like `Qwen2.5-0.5B-Instruct` to follow the instruction reliably),
and building it was deliberately left for later rather than faking a "zero-shot" number from
the wrong architecture.

## `TrainConfig` as a plain dataclass, no YAML config system

Same reasoning as `llmtrace` (project 3): this project has exactly one training run shape.
`LoraSettings`/`TrainConfig` are overridable in code (`TrainConfig(base_model=..., lora=LoraSettings(r=16))`)
without a YAML-parsing layer that would add a dependency and a second way to make the same
mistake (a typo'd key silently ignored) for a config this small.

## W&B left as a manual Colab step, not a `TrainConfig` flag

The portfolio plan mentions Weights & Biases for run tracking. It's deliberately not wired into
`TrainConfig`/`train.py` as a first-class option (e.g. `cfg.report_to`) because this project
already has `llmtrace` (project 3) for exactly this kind of run observability, and adding a
second, parallel tracking system as a built-in config option would duplicate that without a
clear reason -- W&B's free-tier dashboard is still genuinely useful for the GPU training run
itself (loss curves over training steps, which `llmtrace` doesn't do), so it's documented as an
opt-in `TrainingArguments(report_to=["wandb"])` override in `notebooks/train_on_colab.md`
instead of a permanent code path.

## The tiny-model smoke test uses `target_modules=["c_attn"]`, not the config default

`TrainConfig`'s default `target_modules=["q_proj", "v_proj"]` matches Qwen2.5's (and most
modern decoder models') separate query/value projection layer names. GPT-2 (the tiny proxy
model used for the smoke test) combines query, key and value into one `c_attn` layer instead --
`tests/test_train_smoke.py` overrides `target_modules` specifically for that architecture. This
is a real, observed difference between model families, not an inconsistency: LoRA's
`target_modules` has to name actual module attributes on the specific base model, which is
exactly the kind of model-specific detail worth getting right before the real GPU run, not
after.
