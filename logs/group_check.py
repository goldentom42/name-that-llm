"""Check latest_predict.find_groups / assign_in_groups on train.csv (no prompt_id), subsets, an ungrouped set, and timing."""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, 'src'); import latest_predict as lp
sys.path.insert(0, 'src'); from train_pool import split_prompts
t = pd.read_csv('train.csv'); labels = sorted(t.model.unique()); y = t.model.map({l: i for i, l in enumerate(labels)}).values
held = split_prompts(t.prompt_id.values)
def sh(Q, T):
    L = np.log(np.clip(Q, 1e-9, 1)) / T; E = np.exp(L - L.max(1, keepdims=True)); return E / E.sum(1, keepdims=True)
B = (0.1 * np.load('oof_blend.npy') + 0.2 * np.load('val_pool_tfidf_match_weight_1e-06.npy') + 0.2 * sh(np.load('val_ModernBERT-base.npy'), 3)
     + 0.1 * sh(np.load('val_deberta-v3-base.npy'), 3) + 0.4 * sh(np.load('logs/val_qwen_lora_fp16.npy'), 2))
from sklearn.metrics import adjusted_rand_score
def run(idx, name):
    T = time.time(); g = lp.find_groups(t.text.iloc[idx].tolist()); dt = time.time() - T
    base = (B[idx].argmax(1) == y[idx]).mean()
    if g is None: print(f'{name}: gate off, acc {base:.4f} ({dt:.1f}s)'); return
    ok = lp.assign_in_groups(B[idx], g) == y[idx]
    print(f'{name}: ARI {adjusted_rand_score(t.prompt_id.values[idx], g):.3f} acc {base:.4f} -> {ok.mean():.4f} ({dt:.1f}s)'
          + (f' | select {ok[~held].mean():.4f} holdout {ok[held].mean():.4f}' if len(idx) == len(t) else ''))
run(np.arange(len(t)), 'all train.csv')
rng = np.random.RandomState(1); pr = np.unique(t.prompt_id)
for n in (20, 75): run(np.where(np.isin(t.prompt_id, rng.choice(pr, n, replace=False)))[0], f'{n} prompts')
run(t.groupby('prompt_id').sample(1, random_state=0).index.values, 'ungrouped (1 per prompt)')
P = pd.read_parquet('data/open_pool/ntl_open_pool.parquet', columns=['text', 'prompt_canon'])
for n in (500, 3000, 9000):
    T = time.time(); g = lp.find_groups(P.text.sample(n, random_state=0).tolist())
    print(f'{n} random pool texts (ungrouped): {"gate off" if g is None else "GROUPED"}, {time.time() - T:.0f}s')
# A grouped pool set: canonical prompts with 15+ passages, up to 18 each, 150 prompts.
vc = P.prompt_canon.value_counts(); keep = vc[vc >= 15].index[:150]
sub = P[P.prompt_canon.isin(keep)].groupby('prompt_canon').head(18)
g = lp.find_groups(sub.text.tolist()); print(f'grouped pool set ({len(sub)} texts): {"gate off" if g is None else "GROUPED"}' +
    ('' if g is None else f', ARI {adjusted_rand_score(sub.prompt_canon, g):.3f}'))
