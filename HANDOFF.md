# Name That LLM: handoff for GPU training

## Competition
- Kaggle code competition: identify which of 18 LLMs wrote a text. Hidden test, offline notebook, 9h limit.
- 5 submissions per day. Output file: `submission.parquet` with columns `id,model`.
- External public data and pre-trained models are allowed.
- Metric not confirmed; accuracy assumed. Top public score was about 0.70 on 2026-11-06.

## Goal for this machine
Fine-tune a transformer on the open pool, on the RTX 3090 (24 GB). The result must run offline on Kaggle:
upload the weights as a Kaggle dataset, and run inference there on a T4 (14.5 GB) inside the 9h limit.

## Data
- `train.csv`: 2,699 texts, 150 prompts, 18 models. `test.csv` is a 3-row stub.
- `data/open_pool/ntl_open_pool.parquet`: 212,623 passages (`crtproduction/name-that-llm-open-pool`),
  columns include `text`, `model`, `match_weight`, `prompt_canon`, `flag_kb_proxy`.
- No text overlap between the pool and train.csv/test.csv, so use ALL of train.csv as validation.
- The pool's style differs from the competition data (e.g. curly apostrophes), so a pool-only random split will
  overstate performance. Report accuracy on train.csv.

## Results so far (accuracy on train.csv)
| Model | Accuracy |
|---|---|
| TF-IDF/style SVM + Qwen2.5-1.5B features, train.csv only, 5-fold by prompt | 44.5% (0.409 on leaderboard) |
| Linear hashed n-grams on the pool, `match_weight`, alpha 1e-6 | 48.2% |
| 40% pool model + 60% old blend | 53.0% (51.0% on a 30-prompt holdout) |
| DeBERTa-v3-base fine-tuned on train.csv only, fold 0 | 21.3% |

## Findings
- Keep `flag_kb_proxy` rows: dropping them cut the linear model from 48.2% to 44.4%.
- `match_weight` helped the linear model slightly (48.2% vs 47.8%).
- transformers 5.x loads DeBERTa-v3 as fp16 by default, which gave NaN loss. Pass `dtype=torch.float32`.
- DeBERTa's tokenizer drops `…` and collapses repeated spaces. ModernBERT is worth comparing.
- Blend new models with `oof_blend.npy` (old blend) and `val_pool_tfidf_match_weight_1e-06.npy` (pool model),
  both aligned to train.csv rows.

## Unresolved
Whether the hidden test has prompt groups. Grouping lifted the holdout a lot, but the forced diagnostic has not run:
an earlier 0.409 was NOT that diagnostic. `src/predict_diag.py` is the diagnostic; compare it with `src/predict.py`.

## Next steps
1. Train DeBERTa-v3-base and/or ModernBERT-base on the pool, evaluating on train.csv each epoch.
2. Blend with the two saved prediction files above, picking weights on train.csv.
3. Add the model to the Kaggle notebook, offline, with weights attached as a dataset.
