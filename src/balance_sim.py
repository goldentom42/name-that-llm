"""Marginal balancing on train.csv out-of-fold blend probabilities.

The metric is balanced accuracy. If the hidden test has about equal passages per model, pushing the
predicted class shares towards uniform can fix systematic over- and under-prediction. Methods:
  soft   : Sinkhorn-style prior correction, p_ic * w_c renormalised, w chosen so the mean predicted
           probability of every class is 1/K; strength a applies w**a (a=0: none, a=1: full)
  hard   : equal-count assignment (each class gets n/K predictions) by linear_sum_assignment on log p
Checks: select / holdout / all (balanced accuracy) on the current blend, and robustness when the
evaluated set is NOT balanced (class shares drawn from a Dirichlet), where balancing could hurt.

    python src/balance_sim.py
"""
import sys, numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
sys.path.insert(0, 'src'); from train_pool import split_prompts

t = pd.read_csv('train.csv'); labels = sorted(t.model.unique()); K = len(labels)
y = t.model.map({l: i for i, l in enumerate(labels)}).values; held = split_prompts(t.prompt_id.values)


def sh(Q, T):
    L = np.log(np.clip(Q, 1e-9, 1)) / T; E = np.exp(L - L.max(1, keepdims=True)); return E / E.sum(1, keepdims=True)


# Current submission's blend, with the 2-fold OOF predictions of the mixed Qwen.
B = (0.1 * np.load('oof_blend.npy') + 0.2 * np.load('val_pool_tfidf_match_weight_1e-06.npy') + 0.3 * sh(np.load('val_ModernBERT-base.npy'), 3)
     + 0.1 * sh(np.load('val_deberta-v3-base.npy'), 5) + 0.3 * sh(np.load('oof_qwen_mix_full.npy'), 2))


def soft_weights(P, iters=200):
    w = np.ones(P.shape[1])
    for _ in range(iters):
        Q = P * w; Q /= Q.sum(1, keepdims=True)
        w /= Q.mean(0) * P.shape[1]
    return w


def soft(P, a):
    Q = P * soft_weights(P) ** a; return Q / Q.sum(1, keepdims=True)


def hard(P):
    n = len(P); cap = int(np.ceil(n / K))
    cost = np.repeat(np.log(P + 1e-12), cap, axis=1)  # columns: K classes x cap slots
    r, c = linear_sum_assignment(cost, maximize=True)
    pred = np.empty(n, int); pred[r] = c // cap; return pred


def report(name, pred, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    bs = lambda m: balanced_accuracy_score(y[idx][m], pred[m])
    print(f'{name:28s} select {bs(~held[idx]):.4f} holdout {bs(held[idx]):.4f} all {bs(np.ones(len(idx), bool)):.4f} '
          f'| predicted per class min {np.bincount(pred, minlength=K).min()} max {np.bincount(pred, minlength=K).max()}')


print('class counts in train.csv:', np.bincount(y).min(), '-', np.bincount(y).max())
report('none', B.argmax(1))
for a in (0.25, 0.5, 0.75, 1.0):
    report(f'soft a={a}', soft(B, a).argmax(1))
report('hard equal counts', hard(B))

print('\nPer-class recall, none -> soft a=1:')
p0, p1 = B.argmax(1), soft(B, 1).argmax(1)
for c in range(K):
    m = y == c
    print(f'  {labels[c]:24s} {np.mean(p0[m] == c):.2f} -> {np.mean(p1[m] == c):.2f}  (predicted {np.sum(p0 == c):3d} -> {np.sum(p1 == c):3d})')

# Robustness: evaluate on subsets of ~1,350 rows whose class shares are NOT uniform.
rng = np.random.RandomState(0); res = {k: [] for k in ('none', 'soft0.5', 'soft1', 'hard')}
for conc in (50, 10, 3):
    gains = {k: [] for k in res}
    for s in range(20):
        share = rng.dirichlet(np.full(K, conc)); idx = []
        for c in range(K):
            pool = np.where(y == c)[0]; idx += list(rng.choice(pool, min(len(pool), int(round(share[c] * 1350 * 1.0))), replace=False))
        idx = np.array(idx); P = B[idx]; yy = y[idx]
        ba = lambda pred: balanced_accuracy_score(yy, pred)
        base = ba(P.argmax(1))
        gains['none'].append(0); gains['soft0.5'].append(ba(soft(P, 0.5).argmax(1)) - base)
        gains['soft1'].append(ba(soft(P, 1).argmax(1)) - base); gains['hard'].append(ba(hard(P)) - base)
    sizes = 'uniform-ish' if conc == 50 else 'moderately skewed' if conc == 10 else 'strongly skewed'
    print(f'class shares Dirichlet({conc}) [{sizes}]: gain vs none  ' +
          '  '.join(f'{k} {np.mean(v):+.4f} (min {np.min(v):+.4f})' for k, v in gains.items() if k != 'none'))
