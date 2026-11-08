"""Blend transformer predictions with the two saved models; weights picked on 120 prompts, reported on 30.

    python src/blend_pool.py val_ModernBERT-base.npy [val_deberta-v3-base.npy ...]
"""
import itertools, sys, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from train_pool import split_prompts

t = pd.read_csv('train.csv'); labels = sorted(t.model.unique())
y = t.model.map({l: i for i, l in enumerate(labels)}).values
held = split_prompts(t.prompt_id.values)
names = ['oof_blend.npy', 'val_pool_tfidf_match_weight_1e-06.npy'] + sys.argv[1:]
Ps = [np.load(n) for n in names]
for n, P in zip(names, Ps):
    assert P.shape == (len(t), len(labels)), n
    ok = P.argmax(1) == y
    print(f'{n:44s} all {ok.mean():.4f} select {ok[~held].mean():.4f} holdout {ok[held].mean():.4f}')

def sharpen(P, T):
    Q = np.log(np.clip(P, 1e-9, 1)) / T
    Q = np.exp(Q - Q.max(1, keepdims=True)); return Q / Q.sum(1, keepdims=True)

grid = np.arange(0, 1.01, 0.1); temps = (0.5, 1, 2, 3, 5); best = (-1,)
for w in itertools.product(grid, repeat=len(Ps) - 1):
    if sum(w) > 1 + 1e-9: continue
    w = (*w, 1 - sum(w))
    for T in itertools.product(temps, repeat=len(Ps) - 2):   # temperature on each transformer only
        Qs = Ps[:2] + [sharpen(P, x) for P, x in zip(Ps[2:], T)]
        B = sum(a * Q for a, Q in zip(w, Qs))
        s = (B.argmax(1) == y)[~held].mean()
        if s > best[0]: best = (s, w, T, B)
s, w, T, B = best; ok = B.argmax(1) == y
print('weights', dict(zip(names, np.round(w, 2))), 'temps', T)
print(f'blend: select {s:.4f}  holdout {ok[held].mean():.4f}  all {ok.mean():.4f}')
ref = (0.4 * Ps[1] + 0.6 * Ps[0]).argmax(1) == y
print(f'old 40/60 blend for reference: select {ref[~held].mean():.4f} holdout {ref[held].mean():.4f} all {ref.mean():.4f}')
