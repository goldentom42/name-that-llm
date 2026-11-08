# Train fast linear models on the open pool, validate on all of train.csv (no shared texts).
import time, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
p = pd.read_parquet('data/open_pool/ntl_open_pool.parquet'); t = pd.read_csv('train.csv')
labels = sorted(t.model.unique()); l2i = {l: i for i, l in enumerate(labels)}
for f in [c for c in p.columns if c.startswith('flag_')]: print(f, int(p[f].sum()), flush=True)
p = p[~p.flag_near_train_prompt & ~p.flag_train_prompt_regen].reset_index(drop=True)
print('pool used', len(p), 'match_weight', p.match_weight.describe().round(3).to_dict(), flush=True)
ys, yt = p.model.map(l2i).values, t.model.map(l2i).values
fam = np.array([l.split('-')[0] for l in labels])
hv = [HashingVectorizer(analyzer='char', ngram_range=(2, 5), n_features=2**23, alternate_sign=False, lowercase=False),
      HashingVectorizer(ngram_range=(1, 2), n_features=2**22, alternate_sign=False, lowercase=False, token_pattern=r"(?u)\b\w+\b|[^\w\s]")]
T = time.time(); Xs, Xv = [], []
for h in hv:
    a = h.transform(p.text); tf = TfidfTransformer(sublinear_tf=True).fit(a)
    Xs.append(tf.transform(a)); Xv.append(tf.transform(h.transform(t.text))); print('vectorised', a.shape, f'{time.time()-T:.0f}s', flush=True)
Xs, Xv = sp.hstack(Xs).tocsr(), sp.hstack(Xv).tocsr()
for w_name, w in [('uniform', None), ('match_weight', p.match_weight.values)]:
    for alpha in (1e-6, 1e-5):
        m = SGDClassifier(loss='log_loss', alpha=alpha, max_iter=6, tol=None, n_jobs=-1, random_state=0).fit(Xs, ys, sample_weight=w)
        P = m.predict_proba(Xv); a = P.argmax(1)
        np.save(f'val_pool_tfidf_{w_name}_{alpha}.npy', P)
        print(f'{w_name:13s} alpha={alpha}: acc {(a==yt).mean():.4f} family {(fam[a]==fam[yt]).mean():.4f} ({time.time()-T:.0f}s)', flush=True)
