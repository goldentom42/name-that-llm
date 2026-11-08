"""DIAGNOSTIC for one submission: predict.py plus forced prompt-group logic.
No group check and no error fallback: a crash fails the submission, on purpose.

Name That LLM: fit the best blend on train and write submission.parquet.

Runs unchanged locally (Mac/MPS) and in an offline Kaggle notebook (CUDA or CPU).
Blend = 0.4 * SGD log-loss on hashed n-grams, trained on the external open pool
      + 0.6 * (0.7 * TF-IDF/style LinearSVC (softmax of scores*3)
             + 0.3 * LogisticRegression on Qwen2.5-1.5B features
                     (masked mean of hidden states at layers 4 and 20 + token log-prob stats)).
Train LLM features are computed in the same environment as test features, so both
come from the same hardware/precision.

Kaggle: add as inputs the competition data, the Qwen2.5 1.5B (base, transformers) model,
and the crtproduction/name-that-llm-open-pool dataset.
Env overrides: NTL_DATA_DIR, NTL_POOL, NTL_MODEL_DIR, NTL_OUT, NTL_DEVICE, NTL_BATCH.
"""
import glob, os, re, time
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np, pandas as pd, scipy.sparse as sp, torch
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ON_KAGGLE = os.path.isdir('/kaggle/input')
LAYERS, W_LLM, LR_C, SVC_C, SVC_TEMP, STYLE_W = (4, 20), 0.3, 0.003, 0.5, 3, 0.3
W_POOL, POOL_ALPHA = 0.4, 1e-6
N_MODELS, W_CENTER, CLUSTER_DIST, MIN_GROUP = 18, 0.3, 0.9, 5


def find_data_dir():
    if os.environ.get('NTL_DATA_DIR'): return os.environ['NTL_DATA_DIR']
    if not ON_KAGGLE: return '.'
    hits = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/train.csv', recursive=True)
            if os.path.exists(os.path.join(os.path.dirname(p), 'test.csv'))]
    assert hits, 'competition data not found under /kaggle/input'
    return hits[0]


def find_model_dir():
    if os.environ.get('NTL_MODEL_DIR'): return os.environ['NTL_MODEL_DIR']
    if not ON_KAGGLE: return 'Qwen/Qwen2.5-1.5B'
    cfgs = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/config.json', recursive=True)]
    hits = [c for c in cfgs if '1.5b' in c.lower() and 'instruct' not in c.lower()] or \
           [c for c in cfgs if '1.5b' in c.lower()]
    assert hits, f'Qwen2.5-1.5B not found under /kaggle/input (config.json dirs: {cfgs})'
    return sorted(hits)[0]


def find_pool(data_dir):
    if os.environ.get('NTL_POOL'): return os.environ['NTL_POOL']
    if not ON_KAGGLE: return 'data/open_pool/ntl_open_pool.parquet'
    hits = glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True)
    assert hits, 'attach the Name That LLM Open Pool dataset as a notebook input'
    return hits[0]


def pick_device():
    if os.environ.get('NTL_DEVICE'): return os.environ['NTL_DEVICE']
    if torch.cuda.is_available(): return 'cuda'
    if torch.backends.mps.is_available(): return 'mps'
    return 'cpu'


def style(t):
    sents = [s for s in re.split(r'(?<=[.!?])\s+', t) if s]; w = t.split(); n = len(t)
    f = [n, np.log(n), len(w), len(sents), np.mean([len(s) for s in sents]), np.std([len(s) for s in sents]),
         np.mean([len(x) for x in w]), t.count('\n')]
    for ch in ['—', '–', ' - ', '-', ',', ';', ':', '(', '’', "'", '“', '"', '!', '?', '…', '*', '#']:
        f.append(t.count(ch) / n * 1000)
    f += [sum(c.isupper() for c in t) / n, sum(c.isdigit() for c in t) / n, t.endswith('.'),
          len(set(x.lower() for x in w)) / max(len(w), 1)]
    return f


def tfidf_probs(train_text, y, test_text):
    vecs = [TfidfVectorizer(analyzer='char', ngram_range=(1, 6), sublinear_tf=True, min_df=2, lowercase=False, max_features=400000),
            TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, min_df=2, lowercase=False, token_pattern=r"(?u)\b\w+\b|[^\w\s]")]
    for v in vecs: v.fit(train_text)
    S_tr = np.array([style(t) for t in train_text], float); S_te = np.array([style(t) for t in test_text], float)
    sc = StandardScaler().fit(S_tr)
    X = lambda txt, S: sp.hstack([v.transform(txt) for v in vecs] + [sp.csr_matrix(sc.transform(S) * STYLE_W)]).tocsr()
    svc = LinearSVC(C=SVC_C).fit(X(train_text, S_tr), y)
    return softmax(svc.decision_function(X(test_text, S_te)) * SVC_TEMP, 1)


def pool_probs(pool_path, labels, test_text):
    """Linear model on the external open pool (212k passages), weighted by match_weight."""
    pool = pd.read_parquet(pool_path, columns=['text', 'model', 'match_weight'])
    ys = pool.model.map({l: i for i, l in enumerate(labels)}).values
    hv = [HashingVectorizer(analyzer='char', ngram_range=(2, 5), n_features=2**23, alternate_sign=False, lowercase=False),
          HashingVectorizer(ngram_range=(1, 2), n_features=2**22, alternate_sign=False, lowercase=False, token_pattern=r"(?u)\b\w+\b|[^\w\s]")]
    Xs, Xt = [], []
    for h in hv:
        a = h.transform(pool.text); tf = TfidfTransformer(sublinear_tf=True).fit(a)
        Xs.append(tf.transform(a)); Xt.append(tf.transform(h.transform(test_text)))
    clf = SGDClassifier(loss='log_loss', alpha=POOL_ALPHA, max_iter=6, tol=None, n_jobs=-1, random_state=0)
    clf.fit(sp.hstack(Xs).tocsr(), ys, sample_weight=pool.match_weight.values)
    return clf.predict_proba(sp.hstack(Xt).tocsr())


class LLMFeaturizer:
    def __init__(self, model_dir, device):
        self.device = device
        dtype = torch.float32 if device == 'cpu' else torch.float16
        key = 'dtype' if int(transformers.__version__.split('.')[0]) >= 5 else 'torch_dtype'
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = 'right'
        self.m = AutoModelForCausalLM.from_pretrained(model_dir, **{key: dtype}).to(device).eval()

    @torch.no_grad()
    def __call__(self, texts, batch_size):
        texts = list(texts); n = len(texts)
        order = np.argsort([len(t) for t in texts])[::-1]  # longest first: OOM shows up immediately
        out = np.zeros((n, self.m.config.hidden_size * len(LAYERS) + 6), np.float32); t0 = time.time()
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            b = self.tok([texts[i] for i in idx], return_tensors='pt', padding=True).to(self.device)
            # Run the transformer body only; the ~152k-entry vocab head is applied one text at a
            # time below, because full logits for a batch need many GB of GPU memory.
            o = self.m.model(**b, output_hidden_states=True)
            mask = b.attention_mask.unsqueeze(-1).float()
            pooled = [(o.hidden_states[l].float() * mask).sum(1) / mask.sum(1) for l in LAYERS]
            for j, i in enumerate(idx):
                L = int(b.attention_mask[j].sum())  # right-padded, so real tokens are 0..L-1
                lg = self.m.lm_head(o.last_hidden_state[j, :L - 1]).float()
                tgt = b.input_ids[j, 1:L, None]
                v = torch.log_softmax(lg, -1).gather(1, tgt)[:, 0].cpu().numpy()
                r = (lg > lg.gather(1, tgt)).sum(-1, dtype=torch.int32).cpu().numpy()  # rank of the true token
                ppl = [v.mean(), v.std(), v.min(), np.median(v), (r == 0).mean(), (r < 5).mean()]
                out[i] = np.concatenate([p[j].cpu().numpy() for p in pooled] + [np.array(ppl, np.float32)])
            if (s // batch_size) % 50 == 0: print(f'  llm feats {min(s + batch_size, n)}/{n} {time.time() - t0:.0f}s', flush=True)
        assert np.isfinite(out).all(), 'non-finite LLM features'
        return out


def find_groups(texts):
    """Cluster texts by topic; clusters above 18 texts are split."""
    n = len(texts)
    if n < 2 * MIN_GROUP: return None  # the visible 3-row run
    Z = TfidfVectorizer(sublinear_tf=True, stop_words='english').fit_transform(texts).toarray()
    labels = AgglomerativeClustering(n_clusters=None, distance_threshold=CLUSTER_DIST, metric='cosine', linkage='average').fit_predict(Z)
    next_label = labels.max() + 1
    for g in np.where(np.bincount(labels) > N_MODELS)[0]:
        i = np.where(labels == g)[0]; k = int(np.ceil(len(i) / N_MODELS))
        sub = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average').fit_predict(Z[i])
        labels[i] = np.where(sub == 0, g, next_label + sub - 1); next_label += k - 1
    return labels


def center_by(X, groups):
    Z = X.copy()
    for g in np.unique(groups): Z[groups == g] -= Z[groups == g].mean(0)
    return Z


def grouped_predict(P, Xq_tr, y, prompt_ids, Xq_te, groups):
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=LR_C, max_iter=3000)).fit(center_by(Xq_tr, prompt_ids), y)
    pc = lr.predict_proba(center_by(Xq_te, groups))
    big = np.bincount(groups)[groups] >= MIN_GROUP
    B = P.copy(); B[big] = (1 - W_CENTER) * P[big] + W_CENTER * pc[big]
    pred = B.argmax(1)
    for g in np.unique(groups):
        i = np.where(groups == g)[0]
        r, c = linear_sum_assignment(np.log(B[i] + 1e-9), maximize=True)
        pred[i[r]] = c
    return pred, B


def main():
    data_dir, model_dir, device = find_data_dir(), find_model_dir(), pick_device()
    out_path = os.environ.get('NTL_OUT', '/kaggle/working/submission.parquet' if ON_KAGGLE else 'submission.parquet')
    batch = int(os.environ.get('NTL_BATCH', 16 if device == 'cuda' else 4 if device == 'mps' else 2))
    print(f'data={data_dir} model={model_dir} device={device} batch={batch} out={out_path}', flush=True)

    train, test = pd.read_csv(f'{data_dir}/train.csv'), pd.read_csv(f'{data_dir}/test.csv')
    labels = sorted(train.model.unique())
    y = train.model.map({l: i for i, l in enumerate(labels)}).values
    print(f'train {len(train)} test {len(test)} classes {len(labels)}', flush=True)

    t = time.time(); p_svc = tfidf_probs(train.text, y, test.text)
    print(f'tfidf svc done {time.time() - t:.0f}s', flush=True)

    feat = LLMFeaturizer(model_dir, device)
    cache = None if ON_KAGGLE else f'feats_train_{device}_{os.path.basename(model_dir.rstrip("/"))}.npz'
    if cache and os.path.exists(cache):
        Xq_tr = np.load(cache)['X']; assert len(Xq_tr) == len(train), 'stale cache'
    else:
        Xq_tr = feat(train.text, batch)
        if cache: np.savez(cache, X=Xq_tr, y=y)
    Xq_te = feat(test.text, batch)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=LR_C, max_iter=3000)).fit(Xq_tr, y)
    p_llm = lr.predict_proba(Xq_te)
    print(f'qwen LR done {time.time() - t:.0f}s', flush=True)

    P = (1 - W_LLM) * p_svc + W_LLM * p_llm
    p_pool = pool_probs(find_pool(data_dir), labels, test.text)
    print(f'pool model done {time.time() - t:.0f}s', flush=True)
    P = W_POOL * p_pool + (1 - W_POOL) * P
    pred = P.argmax(1)
    groups = find_groups(test.text.tolist())
    print('DIAG groups:', 'none (small test)' if groups is None else f'{len(np.unique(groups))} clusters', flush=True)
    if groups is not None:
        pred, P = grouped_predict(P, Xq_tr, y, train.prompt_id.values, Xq_te, groups)
    sub = pd.DataFrame({'id': test.id.astype(str), 'model': [labels[i] for i in pred]})
    sub.to_parquet(out_path, index=False)
    np.save(out_path.replace('.parquet', '_probs.npy'), P)
    print(sub.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
