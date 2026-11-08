import hashlib, numpy as np, pandas as pd
from scipy.special import softmax
from scipy.optimize import linear_sum_assignment as lsa
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
d = pd.read_csv('train.csv'); te = pd.read_csv('test.csv'); labels = sorted(d.model.unique())
C = np.load('feats_train_mps_Qwen2.5-1.5B.npz'); X, y = C['X'], C['y']
folds = list(GroupKFold(5).split(d, y, d.prompt_id))
svc_p = softmax(np.load('oof_cheap.npy', allow_pickle=True).item()['svc'] * 3, 1)
P = np.zeros_like(svc_p)
for tr, va in folds:
    P[va] = make_pipeline(StandardScaler(), LogisticRegression(C=0.003, max_iter=3000)).fit(X[tr], y[tr]).predict_proba(X[va])
B = 0.7 * svc_p + 0.3 * P
np.save('oof_blend.npy', B)
print('blend no assignment', round((B.argmax(1) == y).mean(), 4))

def assign(groups):
    pred = B.argmax(1).copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        r, c = lsa(-np.log(B[idx] + 1e-9)[:, :] * -1 * -1)
        pred[idx[r]] = c
    return pred
def assign_max(groups):
    pred = B.argmax(1).copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        r, c = lsa(np.log(B[idx] + 1e-9), maximize=True)
        pred[idx[r]] = c
    return pred
true_g = d.prompt_id.astype('category').cat.codes.values
print('oracle prompt groups + assignment', round((assign_max(true_g) == y).mean(), 4))

# non-oracle: cluster each validation fold by topic, no prompt_id used
for name, Z in [('tfidf word', None)]:
    pred = B.argmax(1).copy(); aris = []
    for _, va in folds:
        Z = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2).fit_transform(d.text.iloc[va]).toarray()
        k = len(np.unique(true_g[va]))
        cl = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average').fit_predict(Z)
        aris.append(adjusted_rand_score(true_g[va], cl))
        for g in np.unique(cl):
            i = va[cl == g]
            if len(i) > 18:  # oversized cluster: keep plain argmax
                continue
            r, c = lsa(np.log(B[i] + 1e-9), maximize=True); pred[i[r]] = c
    print(f'{name} clustering ARI {np.mean(aris):.3f} | acc with assignment {(pred == y).mean():.4f}')

# leak checks
for fn in (hashlib.md5, hashlib.sha1, hashlib.sha256):
    print(fn.__name__, 'id is prefix of text hash:', (d.text.map(lambda t: fn(t.encode()).hexdigest()[:12]) == d.id).mean())
print('duplicate texts in train', d.text.duplicated().sum(), '| id len', d.id.str.len().unique())
idn = d.id.map(lambda s: int(s, 16))
print('id-file order monotonic', idn.is_monotonic_increasing)
tv = TfidfVectorizer(sublinear_tf=True, stop_words='english').fit(d.text)
sim = (tv.transform(te.text) @ tv.transform(d.text).T).toarray()
for i in range(len(te)):
    j = sim[i].argmax(); print('test', i, 'top sim', sim[i].max().round(3), d.prompt_id[j], '|', te.text[i][:60], '||', d.text[j][:60])
print('length by model (mean chars):'); print(d.groupby('model').text.apply(lambda s: s.str.len().mean()).round(0).to_string())
