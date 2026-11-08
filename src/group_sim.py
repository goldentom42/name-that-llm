"""Simulate the prompt-group step on train.csv without using prompt_id.

Clusters texts by topic (TF-IDF on the evaluated texts only), splits clusters above 18 texts,
then assigns labels within each cluster with the Hungarian algorithm (each model at most once).
Also checks the gate statistic on an ungrouped set (one text per prompt), where grouping must switch off.

    python src/group_sim.py
"""
import sys, numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, 'src'); from train_pool import split_prompts

N = 18
t = pd.read_csv('train.csv'); labels = sorted(t.model.unique()); y = t.model.map({l: i for i, l in enumerate(labels)}).values
held = split_prompts(t.prompt_id.values); g_true = t.prompt_id.astype('category').cat.codes.values


def sh(Q, T):
    L = np.log(np.clip(Q, 1e-9, 1)) / T; E = np.exp(L - L.max(1, keepdims=True)); return E / E.sum(1, keepdims=True)


B = (0.1 * np.load('oof_blend.npy') + 0.2 * np.load('val_pool_tfidf_match_weight_1e-06.npy') + 0.2 * sh(np.load('val_ModernBERT-base.npy'), 3)
     + 0.1 * sh(np.load('val_deberta-v3-base.npy'), 3) + 0.4 * sh(np.load('logs/val_qwen_lora_fp16.npy'), 2))


def cluster(texts, mode, thr):
    Z = TfidfVectorizer(sublinear_tf=True, stop_words='english').fit_transform(texts).toarray()
    if mode == 'count':
        lab = AgglomerativeClustering(n_clusters=max(1, round(len(texts) / N)), metric='cosine', linkage='average').fit_predict(Z)
    else:
        lab = AgglomerativeClustering(n_clusters=None, distance_threshold=thr, metric='cosine', linkage='average').fit_predict(Z)
    nxt = lab.max() + 1
    for g in np.where(np.bincount(lab) > N)[0]:  # split oversized clusters
        i = np.where(lab == g)[0]; k = int(np.ceil(len(i) / N))
        sub = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average').fit_predict(Z[i])
        lab[i] = np.where(sub == 0, g, nxt + sub - 1); nxt += k - 1
    return lab


def assign(P, groups, min_group):
    pred = P.argmax(1).copy()
    for g in np.unique(groups):
        i = np.where(groups == g)[0]
        if len(i) < min_group: continue
        r, c = linear_sum_assignment(np.log(P[i] + 1e-9), maximize=True); pred[i[r]] = c
    return pred


gate = lambda lab: (np.bincount(lab)[lab] >= N // 2).mean()  # share of texts in clusters of 9+

print(f'no grouping: select {(B.argmax(1) == y)[~held].mean():.4f} holdout {(B.argmax(1) == y)[held].mean():.4f} all {(B.argmax(1) == y).mean():.4f}')
configs = [('count', None)] + [('thr', x) for x in (0.7, 0.8, 0.85, 0.9, 0.95)]
for mode, thr in configs:
    lab = cluster(t.text.tolist(), mode, thr)
    line = f'{mode} {thr}: clusters {lab.max() + 1} ARI {adjusted_rand_score(g_true, lab):.3f} gate {gate(lab):.2f}'
    for mg in (1, 5, 9):
        ok = assign(B, lab, mg) == y
        line += f' | min{mg}: sel {ok[~held].mean():.4f} hold {ok[held].mean():.4f} all {ok.mean():.4f}'
    print(line, flush=True)

# Robustness: smaller test sets (random subsets of prompts), and an ungrouped set for the gate.
rng = np.random.RandomState(0); prompts = np.unique(t.prompt_id)
for mode, thr in configs:
    res = []
    for n_p in (20, 50, 100):
        for s in range(3):
            keep = np.isin(t.prompt_id, rng.choice(prompts, n_p, replace=False)); idx = np.where(keep)[0]
            lab = cluster(t.text.iloc[idx].tolist(), mode, thr)
            res.append(((assign(B[idx], lab, 5) == y[idx]).mean() - (B[idx].argmax(1) == y[idx]).mean(), gate(lab)))
    one = t.groupby('prompt_id').sample(1, random_state=0).index.values  # 150 texts, all different prompts
    lab1 = cluster(t.text.iloc[one].tolist(), mode, thr)
    print(f'{mode} {thr}: subsets gain mean {np.mean([r[0] for r in res]):+.4f} min {np.min([r[0] for r in res]):+.4f}, '
          f'gate min {np.min([r[1] for r in res]):.2f} | ungrouped set: gate {gate(lab1):.2f}, '
          f'assign gain {(assign(B[one], lab1, 5) == y[one]).mean() - (B[one].argmax(1) == y[one]).mean():+.4f}', flush=True)
