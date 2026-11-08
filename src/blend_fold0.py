import pandas as pd, numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline, make_union
d = pd.read_csv('train.csv'); labels = sorted(d.model.unique())
y = d.model.map({l: i for i, l in enumerate(labels)}).values
tr, va = list(GroupKFold(5).split(d, y, d.prompt_id))[0]
p = make_pipeline(make_union(
    TfidfVectorizer(analyzer='char_wb', ngram_range=(1,5), sublinear_tf=True, min_df=2, lowercase=False),
    TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=2)), LogisticRegression(C=10, max_iter=3000))
p.fit(d.text.iloc[tr], y[tr]); pt = p.predict_proba(d.text.iloc[va])
pd_ = softmax(np.load('oof_fold0_deberta-v3-base.npy'), 1)
fam = np.array([l.split('-')[0] for l in labels])
for name, pr in [('tfidf', pt), ('deberta', pd_)] + [(f'blend w_deb={w}', (1-w)*pt + w*pd_) for w in (0.2, 0.3, 0.5)]:
    a = pr.argmax(1); print(f'{name:16s} acc {(a==y[va]).mean():.4f}  family {(fam[a]==fam[y[va]]).mean():.4f}')
