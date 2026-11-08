import time, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
p = pd.read_parquet('data/open_pool/ntl_open_pool.parquet'); t = pd.read_csv('train.csv')
labels = sorted(t.model.unique()); l2i = {l: i for i, l in enumerate(labels)}
ys, yt = p.model.map(l2i).values, t.model.map(l2i).values
fam = np.array([l.split('-')[0] for l in labels]); old = np.load('oof_blend.npy')
hv = [HashingVectorizer(analyzer='char', ngram_range=(2, 5), n_features=2**23, alternate_sign=False, lowercase=False),
      HashingVectorizer(ngram_range=(1, 2), n_features=2**22, alternate_sign=False, lowercase=False, token_pattern=r"(?u)\b\w+\b|[^\w\s]")]
T = time.time(); Xs, Xv = [], []
for h in hv:
    a = h.transform(p.text); tf = TfidfTransformer(sublinear_tf=True).fit(a)
    Xs.append(tf.transform(a)); Xv.append(tf.transform(h.transform(t.text)))
Xs, Xv = sp.hstack(Xs).tocsr(), sp.hstack(Xv).tocsr(); w = p.match_weight.values; print(f'vectorised {time.time()-T:.0f}s', flush=True)
nop = (~p.flag_kb_proxy).values
for name, m_, alpha in [('all', None, 3e-7), ('all', None, 1e-7), ('no_proxy', nop, 1e-6), ('no_proxy', nop, 3e-7)]:
    X, Y, W = (Xs, ys, w) if m_ is None else (Xs[m_], ys[m_], w[m_])
    c = SGDClassifier(loss='log_loss', alpha=alpha, max_iter=6, tol=None, n_jobs=-1, random_state=0).fit(X, Y, sample_weight=W)
    P = c.predict_proba(Xv); P = P / P.sum(1, keepdims=True); np.save(f'val_pool_{name}_{alpha}.npy', P)
    bl = ' '.join(f'w{b}={((b*P+(1-b)*old).argmax(1)==yt).mean():.4f}' for b in (0.5, 0.7, 0.85))
    print(f'{name:9s} alpha={alpha}: acc {(P.argmax(1)==yt).mean():.4f} family {(fam[P.argmax(1)]==fam[yt]).mean():.4f} | blend-with-old {bl} ({time.time()-T:.0f}s)', flush=True)
