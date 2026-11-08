# CV check on the feature cache written by src/predict.py (batched, masked pooling)
import sys, numpy as np, pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
d = pd.read_csv('train.csv'); C = np.load(sys.argv[1]); X, y = C['X'], C['y']
folds = list(GroupKFold(5).split(d, y, d.prompt_id))
svc_p = softmax(np.load('oof_cheap.npy', allow_pickle=True).item()['svc'] * 3, 1)
P = np.zeros_like(svc_p)
for tr, va in folds:
    P[va] = make_pipeline(StandardScaler(), LogisticRegression(C=0.003, max_iter=3000)).fit(X[tr], y[tr]).predict_proba(X[va])
print(f'qwen alone {(P.argmax(1)==y).mean():.4f} | blend {((0.7*svc_p+0.3*P).argmax(1)==y).mean():.4f}  (unbatched reference: 0.4161 / 0.4450)')
