import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment as lsa
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
d = pd.read_csv('train.csv')
C = np.load('feats_train_mps_Qwen2.5-1.5B.npz'); X, y = C['X'], C['y']
g = d.prompt_id.astype('category').cat.codes.values
folds = list(GroupKFold(5).split(d, y, g)); old = np.load('oof_blend.npy')
def center(X):
    Z = X.copy()
    for k in np.unique(g): Z[g == k] -= Z[g == k].mean(0)
    return Z
def oof(X, C):
    P = np.zeros((len(d), 18))
    for tr, va in folds:
        P[va] = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X[tr], y[tr]).predict_proba(X[va])
    return P
def hung(P):
    pred = P.argmax(1).copy()
    for k in np.unique(g):
        i = np.where(g == k)[0]; r, c = lsa(np.log(P[i] + 1e-9), maximize=True); pred[i[r]] = c
    return (pred == y).mean()
Z = center(X)
print('old blend: plain', round((old.argmax(1) == y).mean(), 4), 'assigned', round(hung(old), 4))
for C_ in (0.003, 0.01, 0.03):
    P = oof(Z, C_); np.save(f'oof_center_{C_}.npy', P)
    line = f'centred qwen C={C_}: plain {(P.argmax(1)==y).mean():.4f} assigned {hung(P):.4f}'
    for w in (0.3, 0.5, 0.7):
        B = (1 - w) * old + w * P; line += f' | w{w} assigned {hung(B):.4f}'
    print(line, flush=True)
