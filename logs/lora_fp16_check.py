"""fp16 merged-LoRA inference (latest_predict.lora_probs) vs the bf16 training-time predictions on train.csv."""
import sys, time, numpy as np, pandas as pd, torch, transformers
sys.path.insert(0, 'src'); import latest_predict as lp
from train_pool import split_prompts
t = pd.read_csv('train.csv'); labels = sorted(t.model.unique()); y = t.model.map({l: i for i, l in enumerate(labels)}).values
held = split_prompts(t.prompt_id.values); n = int(sys.argv[1]) if len(sys.argv) > 1 else len(t)
T = time.time(); P = lp.lora_probs('Qwen/Qwen2.5-1.5B', 'models/Qwen2.5-1.5B-lora-full', labels, t.text[:n], 'cuda', 1.0)
dt = time.time() - T
ref = np.load('val_Qwen2.5-1.5B-lora-full.npy')[:n]
print(f'transformers {transformers.__version__}: {n} rows in {dt:.0f}s; argmax agreement {(P.argmax(1) == ref.argmax(1)).mean():.4f}; '
      f'max |dp| {np.abs(P - ref).max():.4f}; acc fp16 {(P.argmax(1) == y[:n]).mean():.4f} vs bf16 {(ref.argmax(1) == y[:n]).mean():.4f}')
if n == len(t):
    np.save('logs/val_qwen_lora_fp16.npy', P)
    sh = lambda Q, T: (lambda L: np.exp(L - L.max(1, keepdims=True)) / np.exp(L - L.max(1, keepdims=True)).sum(1, keepdims=True))(np.log(np.clip(Q, 1e-9, 1)) / T)
    old, pool = np.load('oof_blend.npy'), np.load('val_pool_tfidf_match_weight_1e-06.npy')
    mb, deb = np.load('val_ModernBERT-base.npy'), np.load('val_deberta-v3-base.npy')
    B = 0.1 * old + 0.2 * pool + 0.2 * sh(mb, 3) + 0.1 * sh(deb, 3) + 0.4 * sh(P, 2); ok = B.argmax(1) == y
    print(f'blend with fp16 Qwen: select {ok[~held].mean():.4f} holdout {ok[held].mean():.4f} all {ok.mean():.4f}')
