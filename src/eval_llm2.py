import numpy as np, pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
d = pd.read_csv('train.csv'); labels = sorted(d.model.unique())
y = d.model.map({l: i for i, l in enumerate(labels)}).values
folds = list(GroupKFold(5).split(d, y, d.prompt_id))
F = np.load('feats_Qwen2.5-1.5B.npz'); svc_p = softmax(np.load('oof_cheap.npy', allow_pickle=True).item()['svc'] * 3, 1)
fam = np.array([l.split('-')[0] for l in labels])
def cv(X, C):
    P = np.zeros((len(d), len(labels)))
    for tr, va in folds:
        P[va] = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X[tr], y[tr]).predict_proba(X[va])
    return P
for name, X in [('l4', np.hstack([F['l4'], F['ppl']])), ('l4+l20', np.hstack([F['l4'], F['l20'], F['ppl']]))]:
    for C in (0.001, 0.003, 0.01):
        P = cv(X, C); b = (0.7 * svc_p + 0.3 * P).argmax(1)
        fold_acc = [(b[va] == y[va]).mean() for _, va in folds]
        print(f'{name:7s} C={C}: alone {(P.argmax(1)==y).mean():.4f} | blend w0.3 {(b==y).mean():.4f} family {(fam[b]==fam[y]).mean():.4f} | per-fold {np.round(fold_acc,3)}', flush=True)
