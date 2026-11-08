import pandas as pd, re, sys
pd.set_option('display.width', 200, 'display.max_columns', 40, 'display.max_colwidth', 70)
P = 'data/open_pool/'
p = pd.read_parquet(P + 'ntl_open_pool.parquet'); print(p.shape); print(p.dtypes); print(p.head(3).T)
lab = [c for c in p.columns if c in ('model', 'label')][0]
print(p[lab].value_counts().to_string())
t = pd.read_csv('train.csv'); te = pd.read_csv('test.csv')
txt = [c for c in p.columns if c in ('text', 'passage')][0]
norm = lambda s: s.str.lower().str.replace(r'\W+', ' ', regex=True).str.strip()
ps = set(norm(p[txt]))
print('train exact in pool', t.text.isin(set(p[txt])).sum(), '| train normalised in pool', norm(t.text).isin(ps).sum(), '| test in pool', norm(te.text).isin(ps).sum())
print('pool dup texts', p[txt].duplicated().sum())
print('chars', p[txt].str.len().describe().round(0).to_dict()); print('train chars', t.text.str.len().describe().round(0).to_dict())
print('curly apostrophe rate by label (pool vs train):')
r = pd.DataFrame({'pool': p.groupby(lab)[txt].apply(lambda s: s.str.contains('’').mean()), 'train': t.groupby('model').text.apply(lambda s: s.str.contains('’').mean())}).round(3); print(r.to_string())
print(pd.read_csv(P + 'sources.csv').head(12).to_string())
