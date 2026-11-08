import sys, numpy as np, pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
d = pd.read_csv('train.csv'); labels = sorted(d.model.unique())
y = d.model.map({l: i for i, l in enumerate(labels)}).values
folds = list(GroupKFold(5).split(d, y, d.prompt_id))
F = np.load(sys.argv[1]); ppl = F['ppl']
svc = np.load('oof_cheap.npy', allow_pickle=True).item()['svc']; svc_p = softmax(svc * 3, 1)
print('tfidf svc alone', (svc.argmax(1) == y).mean().round(4))
def cv(X, C):
    P = np.zeros((len(d), len(labels)))
    for tr, va in folds:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X[tr], y[tr]); P[va] = m.predict_proba(X[va])
    return P
print('perplexity feats only', (cv(ppl, 1).argmax(1) == y).mean().round(4))
best = None
for k in [k for k in F.files if k.startswith('l')]:
    for C in (0.01, 0.05):
        P = cv(np.hstack([F[k], ppl]), C); acc = (P.argmax(1) == y).mean()
        bl = {w: ((1 - w) * svc_p + w * P).argmax(1) for w in (0.3, 0.5, 0.7)}
        print(f'{k} C={C}: alone {acc:.4f} | blend ' + ' '.join(f'w{w}={(b == y).mean():.4f}' for w, b in bl.items()), flush=True)
