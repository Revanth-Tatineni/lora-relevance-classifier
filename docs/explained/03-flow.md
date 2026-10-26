# How the pipeline actually flows

## 1. Building the dataset

```
python -m lora_relevance_classifier.build_dataset --source .../rag-qa-api/evals/dataset.jsonl

load_source_questions(path)
    -> reads each JSONL line; a row with answer != null becomes answerable=True with its
       evidence list; a row with answer == null (rag-qa-api's 5 "none-*" questions) becomes
       answerable=False with an empty evidence list

build_pairs(questions, hard_negatives_per_question=2, seed=0)
    for each answerable question:
        -> one PairExample(label=1, pair_kind="positive") per real evidence snippet
        -> up to 2 PairExample(label=0, pair_kind="hard_negative"), each paired with a
           DIFFERENT question's real evidence (never its own)
    for each unanswerable question:
        -> one PairExample(label=0, pair_kind="no_context_negative"), paired with a randomly
           chosen piece of real evidence from elsewhere in the corpus

split_pairs(pairs, val_fraction=0.2, seed=0)
    -> groups by source_id, shuffles the IDs, takes the first ~20% of IDs entirely into val

save_pairs(train, "data/train.jsonl"); save_pairs(val, "data/val.jsonl")
```

Run for real against `rag-qa-api`'s actual data in this session: 27 source questions became 77
pairs, split into 61 train / 16 val.

## 2. Running the word-overlap baseline (no model, no GPU)

```
lora-relevance-classifier compare --val-file data/val.jsonl

for each pair in val:
    word_overlap_predict(pair.question, pair.context, threshold=0.3)
        -> tokenize both into lowercase word sets, predict 1 if the overlap ratio (relative to
           the question's word count) clears the threshold, else 0

compute_metrics(labels, predictions) -> accuracy, precision, recall, f1, false_positive_rate
```

Run for real in this session: accuracy 0.625, precision 1.0, recall 0.143, f1 0.25, FPR 0.0 --
see `00-overview.md` for what that result actually means.

## 3. Training the LoRA adapter (needs the ML stack; needs a GPU for the real target model)

```
lora-relevance-classifier train --base-model Qwen/Qwen2.5-0.5B --epochs 5

build_model_and_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    base_model = AutoModelForSequenceClassification.from_pretrained(cfg.base_model, num_labels=2)
        -> attaches a NEW, randomly-initialized classification head (see 01-architecture.md
           for why this matters)
    lora_config = LoraConfig(task_type=SEQ_CLS, r=cfg.lora.r, target_modules=cfg.lora.target_modules, ...)
    model = get_peft_model(base_model, lora_config)
        -> freezes the base model's weights; only the LoRA adapter matrices (and the new
           classification head) are trainable

train(cfg, train_pairs, val_pairs):
    train_ds, val_ds = tokenized HF Datasets (question + context as a sentence pair, padded/
                                               truncated to cfg.max_length)
    Trainer(model, TrainingArguments(...), train_ds, val_ds, compute_metrics=...).train()
        -> standard supervised fine-tuning loop; compute_metrics (wired to this project's own
           metrics.compute_metrics) runs at the end of every epoch, via eval_strategy="epoch"
    model.save_pretrained(adapter_path); tokenizer.save_pretrained(adapter_path)
        -> only the small LoRA adapter weights are saved, not a full copy of the base model
```

This exact sequence was run for real on CPU with a tiny proxy model in
`tests/test_train_smoke.py` -- the mechanics are proven; only the real target model and a real
GPU session haven't been run yet (`notebooks/train_on_colab.md` is the script for when they
are).

## 4. Comparing the trained adapter against the baseline

```
lora-relevance-classifier compare --val-file data/val.jsonl --adapter-dir runs/lora-adapter/adapter

load_adapter_model(adapter_path, cfg):
    base_model, tokenizer = load_base_model(cfg)        # fresh base model + tokenizer
    model = PeftModel.from_pretrained(base_model, adapter_path)   # applies the saved adapter

evaluate_pairs(model, tokenizer, pairs, max_length):
    predict(...) runs one forward pass per pair (no gradient, @torch.no_grad()), argmax over
        the two logits
    compute_metrics(labels, predictions)   # the SAME function the baseline used
```

Using `compute_metrics` for both the baseline and the adapter is deliberate: the two numbers in
the README's comparison table are guaranteed to mean the same thing, computed the same way,
because they go through the identical function.
