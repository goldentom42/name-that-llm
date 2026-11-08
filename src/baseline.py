import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline, make_union
d = pd.read_csv('train.csv')
y, g = d.model.values, d.prompt_id.values
oof = np.empty(len(d), dtype=object)
for tr, va in GroupKFold(5).split(d, y, g):
    p = make_pipeline(make_union(
        TfidfVectorizer(analyzer='char_wb', ngram_range=(1,5), sublinear_tf=True, min_df=2, lowercase=False),
        TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=2)),
        LogisticRegression(C=10, max_iter=3000))
    p.fit(d.text.iloc[tr], y[tr]); oof[va] = p.predict(d.text.iloc[va])
print('acc', (oof == y).mean())
fam = lambda s: pd.Series(s).str.split(r'[-\d]', n=1).str[0]
print('family acc', (fam(oof).values == fam(y).values).mean())
print(pd.crosstab(pd.Series(y,name='true'), pd.Series(oof,name='pred')).to_string())
tr, va = list(GroupKFold(5).split(d, y, g))[0]
print('fold0 acc', (oof[va] == y[va]).mean())
