# Name That LLM

Code for the Kaggle code competition **Name That LLM**: given a passage, say which of 18 LLMs wrote it
(5 Claude, 4 Gemini, 5 GPT, 2 GLM, Kimi, Qwen). The metric is balanced accuracy (mean per-class recall).
Inference runs in an offline Kaggle notebook (T4, 9 h limit) and writes `submission.parquet` (`id`, `model`).

**Current best: 0.567 public leaderboard** with `src/latest_predict.py`.

## Leaderboard history

| Script | What changed | train.csv (all / 30-prompt holdout) | Public LB |
|---|---|---|---|
| (earliest, not kept) | TF-IDF/style SVM + Qwen2.5-1.5B features, train.csv only | 44.5% / - | 0.409 |
| `src/latest_predict_0481.py` | + linear model on the open pool (40%) | 53.0% / 48.7% | 0.481 |
| `src/latest_predict_0541.py` | + ModernBERT-base and DeBERTa-v3-base fine-tuned on the pool | 59.9% / 55.6% | 0.541 |
| (not kept as a file) | + Qwen2.5-1.5B LoRA classifier trained on the pool | 62.1% / 58.0% | 0.547 |
| `src/latest_predict_grouped_poolqwen.py` | + gated prompt-group assignment | grouped: 76.5% / 74.1% | 0.547 (gate stayed off) |
| **`src/latest_predict.py`** | Qwen LoRA continued on pool + all of train.csv | 62.9% / 59.1% (2-fold OOF) | **0.567** |

Identical resubmissions move the public LB by up to about +/-0.009, so a single gain smaller than about
0.01 is not evidence.

## Data

- `train.csv`: 2,699 passages, 150 prompts, one passage per model per prompt (17-18 per prompt).
- Hidden test: about 4,000 passages on new prompts, with only a few passages per prompt
  (`test.csv` locally is a 3-row stub).
- Open pool (`crtproduction/name-that-llm-open-pool`, in `data/open_pool/`): 212,623 passages from the same
  18 models, collected through their APIs by the community and merged from about 60 datasets. No text overlaps
  train.csv. Each row has a `match_weight` (how closely it matches the competition style).
- Two further community datasets (`seyitkaangunes/ntl-extra-api-passages`, `abdullah1afify/ntl-generations`,
  in `data/ext/`) turned out to be already in the pool (97% of rows).

## Validation

- Models trained only on the pool are evaluated on all of train.csv, which the pool does not contain.
- `split_prompts` (`src/train_pool.py`) holds out 30 of the 150 prompts. Checkpoints and blend weights are picked
  on the other 120 ("select"); the holdout is report-only and is the best predictor of the leaderboard.
- Anything trained on train.csv itself is validated out-of-fold with `GroupKFold` by `prompt_id`. Every prompt
  has one passage per model, so these folds are also perfectly class-balanced.

## Pipeline (`src/latest_predict.py`)

All components output 18-class probabilities; the blend is a weighted sum (weights and temperatures from
`src/blend_pool.py`, chosen on the selection prompts):

| Weight | Component | Trained on |
|---|---|---|
| 0.1 | Old blend: 0.7 TF-IDF/style LinearSVC + 0.3 logistic regression on Qwen2.5-1.5B hidden-state and log-prob features | train.csv (fitted inside the notebook) |
| 0.2 | SGD log-loss on hashed char 2-5 / word 1-2 n-grams, `match_weight`, alpha 1e-6 | open pool |
| 0.3 | ModernBERT-base classifier, temperature 3 | open pool, 3 epochs (`src/train_pool.py`) |
| 0.1 | DeBERTa-v3-base classifier, temperature 5 | open pool, 2 epochs, max 384 tokens |
| 0.3 | Qwen2.5-1.5B LoRA classifier (last-token head), temperature 2 | pool, then pool + train.csv (below) |

Prompt grouping (`find_groups` / `assign_in_groups`) clusters test texts by topic and, if they look like
complete prompt groups, gives each model at most once per cluster. A gate keeps it off unless half the texts sit
in clusters of 15-18; it stays off on the hidden test, so it changes nothing there.

### Qwen LoRA classifier

1. `src/train_qwen_lora.py Qwen/Qwen2.5-1.5B --rows 0 --epochs 1 --evals 8 --tag=-full`: LoRA r=16, alpha 32,
   on all attention and MLP projections, plus a trained 18-class score head; frozen base in bf16, trainable
   weights in fp32, gradient checkpointing, batch 16, max 384 tokens, lr 1e-4 cosine, class-balanced loss with
   `match_weight` capped at 5. One epoch over the whole pool: 148 min on an RTX 3090.
2. `src/qwen_mix_folds.py --final --repeat 5 --pool_rows 20000 --lr 3e-5`: continue that adapter on all of
   train.csv repeated 5 times mixed with 20,000 fresh pool rows (23 min). The same recipe without `--final` runs
   the 2-fold out-of-fold test that validated it.

On Kaggle the adapter is merged into the base weights by hand (`lora_probs`), in fp16 (the T4 has no bf16),
so peft is not needed. fp16 agrees with bf16 on 98.9% of train.csv, under transformers 4.57 and 5.x.

## Findings

What worked:
- **External data from the same models** is the biggest lever: the pool linear model alone beats everything
  trained on train.csv only (48.2% vs 44.5%).
- **Bigger readers**: Qwen2.5-1.5B LoRA (57.7%) beats ModernBERT-base (54.8%) and DeBERTa-v3-base (51.1%), and
  its errors differ from theirs, so it adds to the blend.
- **In-domain data mixed into training**: continuing the Qwen adapter on train.csv + pool added about 1 point
  locally but **+0.020 on the leaderboard**, while the pool-only Qwen added 2.4 holdout points but only +0.006
  on the leaderboard. The hidden test was produced like train.csv; the pool comes from community API runs with
  other settings, so pool-only gains transfer less than local numbers suggest.
- Temperatures above 1 on the transformer outputs help the blend (they are overconfident).
- `flag_kb_proxy` rows in the pool help (dropping them: 48.2% -> 44.4%); `match_weight` helps slightly.

What did not work:
- **Prompt-group assignment**: +14 points when the test is grouped like train.csv, but the hidden test has only
  a few passages per prompt, where forcing it costs 2-11 points. The gate kept it off (0.547 unchanged).
- **Fine-tuning ModernBERT/DeBERTa on train.csv alone after pool training** (5-fold, lr 1e-5 to 4e-5): no gain
  (best 55.8% vs 54.8%, holdout lower). Mixing train.csv into pool training, as for Qwen, has not been tried.
- **Choosing blend weights for accuracy after grouping**: higher select score, lower holdout (overfitting).
- DeBERTa-v3-base fine-tuned on train.csv only: 21.3%.

Practical notes:
- transformers 5.x loads DeBERTa-v3 in fp16 by default, which gives NaN loss: pass `dtype=torch.float32`.
- Tokenizers saved by transformers 5.x do not load with `AutoTokenizer` on 4.x: build
  `PreTrainedTokenizerFast(tokenizer_file=...)` from `tokenizer.json`.
- A Kaggle dataset holding a single folder is flattened (files at the dataset root); the adapter lookup handles
  both layouts.
- Kaggle does not show the log of the hidden-test scoring run, only of the 3-row commit run.
- The local training machine (RTX 3090, i9-9920X) reset itself twice under heavy load; long runs write logs to
  `logs/` with fsync and log GPU/CPU sensors (`src/sensor_log.py`).

## Kaggle setup

Notebook inputs: the competition data, the Qwen2.5 1.5B base model (`qwen-lm/qwen2.5/transformers/1.5b`),
`crtproduction/name-that-llm-open-pool`, `ogrellier/ntl-pool-checkpoints` (ModernBERT-base/, deberta-v3-base/)
and `ogrellier/ntl-qwen-lora` (Qwen2.5-1.5B-lora-full/, Qwen2.5-1.5B-lora-mix/). GPU T4, internet off.
Paste `src/latest_predict.py` as the notebook code. A run takes about 26 min.

`upload/*/dataset-metadata.json` hold the metadata for those datasets
(`kaggle datasets version -p upload/<name> --dir-mode zip`).

## Reproducing locally

```bash
uv venv --python 3.12 && uv pip install torch transformers peft pandas scikit-learn pyarrow kaggle
python src/train_pool.py answerdotai/ModernBERT-base --epochs 3                    # -> models/, val_*.npy
python src/train_pool.py microsoft/deberta-v3-base --epochs 2 --maxlen 384
python src/train_qwen_lora.py Qwen/Qwen2.5-1.5B --rows 0 --epochs 1 --evals 8 --tag=-full
python src/qwen_mix_folds.py --repeat 5 --pool_rows 20000 --lr 3e-5                # 2-fold OOF test
python src/qwen_mix_folds.py --final --repeat 5 --pool_rows 20000 --lr 3e-5        # deployable adapter
python src/blend_pool.py val_ModernBERT-base.npy val_deberta-v3-base.npy oof_qwen_mix_full.npy
```

Data, model weights and prediction arrays are not in git (see `.gitignore`).

## Next steps

1. Marginal balancing: equalise how often each model is predicted (the metric is balanced accuracy).
2. Mix train.csv into ModernBERT/DeBERTa training, as for Qwen.
3. A larger reader (e.g. Qwen3-8B LoRA classifier) trained with the mixing recipe; needs two T4s or 4-bit
   weights at inference.
