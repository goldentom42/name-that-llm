import re, sys, time, pandas as pd, numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
d = pd.read_csv('train.csv'); labels = sorted(d.model.unique())
y = d.model.map({l: i for i, l in enumerate(labels)}).values
folds = list(GroupKFold(5).split(d, y, d.prompt_id))

def style(t):
    sents = [s for s in re.split(r'(?<=[.!?])\s+', t) if s]; w = t.split(); n = len(t)
    f = [n, np.log(n), len(w), len(sents), np.mean([len(s) for s in sents]), np.std([len(s) for s in sents]),
         np.mean([len(x) for x in w]), t.count('\n')]
    for ch in ['—', '–', ' - ', '-', ',', ';', ':', '(', '’', "'", '“', '"', '!', '?', '…', '*', '#']:
        f.append(t.count(ch) / n * 1000)
    f += [sum(c.isupper() for c in t) / n, sum(c.isdigit() for c in t) / n, t.endswith('.'),
          len(set(x.lower() for x in w)) / max(len(w), 1)]
    return f
S = np.array([style(t) for t in d.text], dtype=float)

VEC = {
 'charwb15': lambda: TfidfVectorizer(analyzer='char_wb', ngram_range=(1,5), sublinear_tf=True, min_df=2, lowercase=False),
 'char16':   lambda: TfidfVectorizer(analyzer='char', ngram_range=(1,6), sublinear_tf=True, min_df=2, lowercase=False, max_features=400000),
 'word12':   lambda: TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=2),
 'word13':   lambda: TfidfVectorizer(ngram_range=(1,3), sublinear_tf=True, min_df=2, lowercase=False, token_pattern=r"(?u)\b\w+\b|[^\w\s]"),
}
def run(name, vecs, clf, use_style=False):
    t = time.time(); P = np.zeros((len(d), len(labels)))
    for tr, va in folds:
        Xs = [v().fit(d.text.iloc[tr]) for v in (VEC[k] for k in vecs)]
        Xtr = [x.transform(d.text.iloc[tr]) for x in Xs]; Xva = [x.transform(d.text.iloc[va]) for x in Xs]
        if use_style:
            sc = StandardScaler().fit(S[tr]); Xtr.append(sp.csr_matrix(sc.transform(S[tr]) * 0.3)); Xva.append(sp.csr_matrix(sc.transform(S[va]) * 0.3))
        m = clf().fit(sp.hstack(Xtr).tocsr(), y[tr]); Xv = sp.hstack(Xva).tocsr()
        P[va] = m.predict_proba(Xv) if hasattr(m, 'predict_proba') else m.decision_function(Xv)
    print(f'{name:40s} acc {(P.argmax(1)==y).mean():.4f}  ({time.time()-t:.0f}s)', flush=True)
    return P
LR = lambda C: (lambda: LogisticRegression(C=C, max_iter=3000))
SVC = lambda C: (lambda: LinearSVC(C=C))
R = {}
R['base'] = run('baseline charwb15+word12 LR10', ['charwb15','word12'], LR(10))
R['style_only'] = run('style features only LR', [], LR(1), True) if False else None
R['base+style'] = run('charwb15+word12+style LR10', ['charwb15','word12'], LR(10), True)
R['char16'] = run('char16+word13 LR10', ['char16','word13'], LR(10))
R['char16+style'] = run('char16+word13+style LR10', ['char16','word13'], LR(10), True)
R['svc'] = run('char16+word13+style SVC0.5', ['char16','word13'], SVC(0.5), True)
R['lr30'] = run('char16+word13+style LR30', ['char16','word13'], LR(30), True)
np.save('oof_cheap.npy', {k: v for k, v in R.items() if v is not None}, allow_pickle=True)
