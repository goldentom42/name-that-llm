# Remove the topic subspace learned from TRAIN prompt means; each validation text is scored alone.
import numpy as np, pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
d = pd.read_csv('train.csv'); C = np.load('feats_train_mps_Qwen2.5-1.5B.npz'); X, y = C['X'], C['y']
g = d.prompt_id.astype('category').cat.codes.values
folds = list(GroupKFold(5).split(d, y, g))
svc_p = softmax(np.load('oof_cheap.npy', allow_pickle=True).item()['svc'] * 3, 1)
def run(r, C_):
    P = np.zeros((len(d), 18))
    for tr, va in folds:
        sc = StandardScaler().fit(X[tr]); A, B = sc.transform(X[tr]), sc.transform(X[va])
        if r:
            M = np.array([A[g[tr] == k].mean(0) for k in np.unique(g[tr])]); M -= M.mean(0)
            U = np.linalg.svd(M, full_matrices=False)[2][:r]
            A, B = A - A @ U.T @ U, B - B @ U.T @ U
        P[va] = LogisticRegression(C=C_, max_iter=3000).fit(A, y[tr]).predict_proba(B)
    return P
best = (0, None)
for r in (0, 10, 30, 60, 100):
    for C_ in (0.003, 0.01):
        P = run(r, C_); bl = {w: ((1 - w) * svc_p + w * P).argmax(1) for w in (0.3, 0.5, 0.7)}
        s = ' '.join(f'w{w}={(b == y).mean():.4f}' for w, b in bl.items())
        print(f'r={r:3d} C={C_}: alone {(P.argmax(1) == y).mean():.4f} | blend {s}', flush=True)
