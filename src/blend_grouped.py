"""Blend weights chosen for accuracy AFTER prompt-group assignment (latest_predict.find_groups on train.csv texts).

    python src/blend_grouped.py val_ModernBERT-base.npy val_deberta-v3-base.npy oof_qwen_mix_full.npy
Same grid as blend_pool.py (weights step 0.1, temperatures on the transformer files); selection on the
120 selection prompts, holdout report-only. Prints the best grouped blend and the plain-argmax optimum's grouped score.
"""
import itertools, sys, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from train_pool import split_prompts
import latest_predict as lp

t = pd.read_csv('train.csv'); labels = sorted(t.model.unique())
y = t.model.map({l: i for i, l in enumerate(labels)}).values
held = split_prompts(t.prompt_id.values)
groups = lp.find_groups(t.text.tolist())
names = ['oof_blend.npy', 'val_pool_tfidf_match_weight_1e-06.npy'] + sys.argv[1:]
Ps = [np.load(n) for n in names]


def sharpen(P, T):
    Q = np.log(np.clip(P, 1e-9, 1)) / T
    Q = np.exp(Q - Q.max(1, keepdims=True)); return Q / Q.sum(1, keepdims=True)


sharp = {(k, T): sharpen(P, T) for k, P in enumerate(Ps) if k >= 2 for T in (0.5, 1, 2, 3, 5)}
grid = np.arange(0, 1.01, 0.1); best_g, best_p = (-1,), (-1,)
for w in itertools.product(grid, repeat=len(Ps) - 1):
    if sum(w) > 1 + 1e-9: continue
    w = (*w, 1 - sum(w))
    for T in itertools.product((0.5, 1, 2, 3, 5), repeat=len(Ps) - 2):
        B = w[0] * Ps[0] + w[1] * Ps[1] + sum(a * sharp[(k + 2, x)] for k, (a, x) in enumerate(zip(w[2:], T)))
        sp = (B.argmax(1) == y)[~held].mean()
        if sp > best_p[0]: best_p = (sp, w, T, B)
        if sp < best_p[0] - 0.03: continue  # far below the plain optimum: skip the slower assignment
        sg = (lp.assign_in_groups(B, groups) == y)[~held].mean()
        if sg > best_g[0]: best_g = (sg, w, T, B)
for name, (s, w, T, B) in (('plain-argmax optimum', best_p), ('grouped optimum', best_g)):
    okp, okg = B.argmax(1) == y, lp.assign_in_groups(B, groups) == y
    print(f'{name}: weights {dict(zip(names, np.round(w, 2)))} temps {T}')
    print(f'   plain   select {okp[~held].mean():.4f} holdout {okp[held].mean():.4f} all {okp.mean():.4f}')
    print(f'   grouped select {okg[~held].mean():.4f} holdout {okg[held].mean():.4f} all {okg.mean():.4f}', flush=True)
