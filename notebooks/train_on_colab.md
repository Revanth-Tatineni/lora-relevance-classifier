# Running the real fine-tune on Colab/Kaggle

This project's code runs and is tested entirely on CPU (`tests/test_train_smoke.py` proves the
pipeline works), but the real target model, `Qwen/Qwen2.5-0.5B`, needs a GPU to fine-tune in a
reasonable amount of time. That run hasn't happened yet (see the README's Status section). This
is what to run once it does.

## 1. Build the dataset (on your own machine, no GPU needed)

```bash
python -m lora_relevance_classifier.build_dataset --source /path/to/rag-qa-api/evals/dataset.jsonl
```

This writes `data/train.jsonl` and `data/val.jsonl`. Upload both to the Colab/Kaggle runtime
(or push this repo to GitHub and `git clone` it inside the notebook — simpler, and keeps the
data versioned with the code that produced it).

## 2. In a Colab notebook with a GPU runtime (Runtime -> Change runtime type -> GPU)

```python
!pip install -q torch transformers peft accelerate datasets

import sys
sys.path.insert(0, "/content/lora-relevance-classifier/src")  # wherever the repo was cloned/uploaded

from pathlib import Path
from lora_relevance_classifier.config import TrainConfig
from lora_relevance_classifier.dataset import load_pairs
from lora_relevance_classifier.train import train

cfg = TrainConfig(base_model="Qwen/Qwen2.5-0.5B", output_dir="/content/runs/lora-adapter")
train_pairs = load_pairs(Path("/content/lora-relevance-classifier/data/train.jsonl"))
val_pairs = load_pairs(Path("/content/lora-relevance-classifier/data/val.jsonl"))

adapter_path = train(cfg, train_pairs, val_pairs)
print(adapter_path)
```

Optional: pass `report_to=["wandb"]` via a custom `TrainingArguments` (not currently exposed
through `TrainConfig` -- see `docs/explained/01-architecture.md` for why W&B was left as a
manual step rather than a built-in flag) if you want Weights & Biases run tracking, per the
original portfolio plan.

## 3. Download the adapter and compare

```python
from google.colab import files
import shutil

shutil.make_archive("/content/adapter", "zip", adapter_path)
files.download("/content/adapter.zip")
```

Back on your own machine, unzip it into `runs/lora-adapter/adapter/` and run:

```bash
lora-relevance-classifier compare --val-file data/val.jsonl --adapter-dir runs/lora-adapter/adapter
```

This prints the word-overlap baseline and the LoRA adapter's metrics side by side. Put those
real numbers into the README's results table in place of `RESULTS_PLACEHOLDER`, and update the
resume line with the real before/after accuracy once they exist.

## Expected cost/time (for the README's cost writeup, once run)

Colab's free-tier T4 GPU and Kaggle's free-tier P100 both fine-tune a model this size (LoRA,
not full fine-tuning) on a dataset this small (77 pairs) in a few minutes, not hours -- record
the actual wall-clock time and note whether a free-tier session was enough, since
`open-source-project-plan.md` explicitly asks for a cost/time writeup as part of what makes
this project stand out.
