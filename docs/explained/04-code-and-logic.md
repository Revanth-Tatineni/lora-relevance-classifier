# Code and logic, file by file

## `dataset.py`

```python
other_evidence = [(oid, s) for oid, s in all_evidence if oid != q.id]
sample_size = min(hard_negatives_per_question, len(other_evidence))
for _oid, snippet in rng.sample(other_evidence, sample_size):
    pairs.append(PairExample(q.question, snippet, 0, q.id, "hard_negative"))
```

`all_evidence` is every `(question_id, evidence_snippet)` pair across every answerable
question, built once before the loop. For each question, `other_evidence` filters out that
question's own id -- the one thing that must never happen is a question's own real evidence
showing up labeled as a *negative* for that same question. `min(hard_negatives_per_question,
len(other_evidence))` guards `rng.sample` from raising when there isn't enough other evidence
to sample from (relevant for small test datasets with only one or two questions, exactly the
shape `tests/test_dataset.py` uses). `rng = random.Random(seed)` -- a seeded, local generator,
not the global `random` module -- is what makes `build_pairs` deterministic given the same
inputs and seed, which is what lets `test_build_pairs_creates_hard_negatives_from_other_questions_evidence`
assert on an exact count rather than "some number greater than zero."

```python
source_ids = sorted({p.source_id for p in pairs})
rng = random.Random(seed)
rng.shuffle(source_ids)
n_val = max(1, round(len(source_ids) * val_fraction))
```

`sorted(...)` before `rng.shuffle(...)` matters for determinism: a Python `set`'s iteration
order isn't guaranteed stable across runs/processes, so sorting first guarantees `shuffle`
always starts from the same ordering given the same `seed`, regardless of set-iteration quirks.
`max(1, round(...))` guarantees at least one source question lands in validation even when
`val_fraction * len(source_ids)` would round down to zero (a small dataset edge case the real
27-question dataset doesn't hit, but a test-sized 2-3-question one could).

## `metrics.py`

```python
paired = list(zip(labels, predictions, strict=True))
tp = sum(1 for lbl, pred in paired if lbl == 1 and pred == 1)
```

`zip(..., strict=True)` raises immediately if `labels` and `predictions` ever have different
lengths -- which would otherwise silently truncate to the shorter one and produce a metric
that's quietly wrong rather than an error. The length check at the top of `compute_metrics`
(`raise ValueError(...)`) already catches this earlier with a clearer message;
`strict=True` is the belt-and-suspenders version for anyone who calls `zip` on these two lists
elsewhere without going through `compute_metrics` first.

Every ratio (`accuracy`, `precision`, `recall`, `f1`, `false_positive_rate`) is computed with
an explicit `if ... else 0.0` guard against dividing by zero -- `test_empty_inputs_do_not_divide_by_zero`
and `test_no_predicted_positives_gives_zero_precision_not_a_crash` are the tests that would fail
if any of those guards were missing (an empty validation set, or a baseline that never predicts
"relevant," are both real situations this task can hit, not contrived edge cases).

## `baseline.py`

```python
_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))
```

Lowercases first, then extracts alphanumeric runs -- so punctuation, case, and word order never
affect the overlap calculation, only which distinct word tokens are present. `word_overlap_score`
divides by the *question's* word count specifically (`len(q_words & c_words) / len(q_words)`,
not the union or the context's count) -- the question is usually much shorter than a retrieved
chunk, so normalizing by its length asks "what fraction of the question's own words show up in
this context," which is the more informative direction for this task than normalizing by the
much-longer context.

## `train.py`

```python
def _to_hf_dataset(pairs: list[PairExample], tokenizer, max_length: int) -> Dataset:
    def _tokenize(batch):
        return tokenizer(
            batch["question"],
            batch["context"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    raw = Dataset.from_dict({"question": [...], "context": [...], "label": [...]})
    return raw.map(_tokenize, batched=True)
```

Passing `tokenizer(batch["question"], batch["context"], ...)` -- two positional sequence
arguments, not one concatenated string -- is what makes this a genuine sentence-pair encoding:
the tokenizer inserts the model's actual pair-separator token between them and sets token-type
boundaries where the architecture supports it, rather than this code having to hand-roll a
separator string.

```python
base_model.config.pad_token_id = tokenizer.pad_token_id
```

Needed specifically because `Qwen2.5-0.5B` (like GPT-2, the smoke test's proxy model) has no
pad token by default -- causal LMs are trained to generate, not to be padded into fixed-length
batches, so one isn't defined until something sets it. Without this line, batching more than
one example together would raise rather than train.

```python
args = TrainingArguments(..., eval_strategy="epoch", save_strategy="no", report_to=[], ...)
```

`save_strategy="no"` -- checkpoints aren't written during training; only the final adapter is
saved explicitly after `trainer.train()` returns (`model.save_pretrained(adapter_path)`).
`report_to=[]` disables automatic experiment-tracker auto-detection (which would otherwise try
to initialize Weights & Biases, Comet, etc. the moment any of those packages happen to be
importable) -- W&B stays an explicit opt-in override on the real GPU run, per
`01-architecture.md`.

## `evaluate.py`

```python
@torch.no_grad()
def predict(model, tokenizer, pairs, max_length) -> list[int]:
    model.eval()
    ...
```

`model.eval()` turns off dropout and any other training-only behavior (this project's LoRA
config does use dropout, `lora_dropout=0.05` by default); `@torch.no_grad()` disables gradient
tracking for the whole function, which both saves memory and is the correct mode for pure
inference -- training code (`train.py`) deliberately never uses this decorator, since it needs
gradients.

## `cli.py`

```python
def _cmd_train(args):
    from lora_relevance_classifier.train import train  # deferred import

    ...


def _cmd_compare(args):
    ...
    if args.adapter_dir:
        from lora_relevance_classifier.evaluate import (
            evaluate_pairs,
            load_adapter_model,
        )  # deferred import
```

Both `torch`/`transformers`/`peft`-dependent imports happen *inside* the command functions,
not at module load time. `lora-relevance-classifier compare` with no `--adapter-dir` (the baseline-only
path `tests/test_cli.py` exercises) never imports `torch` at all -- this is what keeps that one
test fast and what would let someone run the baseline comparison in an environment that hasn't
installed the (much heavier) ML dependencies yet.

## Testing strategy

Same philosophy as projects 1-4: test real logic with real execution wherever the cost of
doing so is reasonable, and isolate the one part that genuinely needs a heavy dependency.
Eighteen of the twenty-two tests here never import `torch`. The remaining four
(`test_train_smoke.py`) do something the other projects' test suites don't: actually run a
real training loop, not just a unit of logic around one. That test is slower (tens of seconds,
dominated by loading `transformers`/`torch` and one epoch over 8 pairs) and needs outbound
network access (to download the tiny proxy model from the HF Hub, cached after the first run) --
both costs were judged worth it for the single highest-value thing a fine-tuning project's test
suite can prove: that the actual training code runs to completion and produces a usable,
reloadable adapter.
