"""Light test of latest_predict.py: Qwen, TF-IDF and pool model stubbed; real checkpoints on CPU, 2 threads."""
import os, sys
sys.path.insert(0, 'src')
import numpy as np, torch
torch.set_num_threads(2)
import latest_predict as lp
assert not torch.cuda.is_available()
rng = np.random.RandomState(0)
def rand_probs(n): P = rng.rand(n, 18); return P / P.sum(1, keepdims=True)
class Stub:
    def __init__(self, *a): pass
    def __call__(self, texts, batch): return np.random.RandomState(len(texts)).normal(size=(len(texts), 8)).astype(np.float32)
lp.tfidf_probs = lambda tr, y, te: rand_probs(len(te))
lp.pool_probs = lambda path, labels, te: rand_probs(len(te))
lp.LLMFeaturizer = Stub
lp.main()
# Kaggle checkpoint lookup, with /kaggle/input faked
lp.ON_KAGGLE = True; real_glob = lp.glob.glob
lp.glob.glob = lambda pat, recursive=False: ['/kaggle/input/ntl-ckpt/ModernBERT-base/config.json', '/kaggle/input/ntl-ckpt/deberta-v3-base/config.json',
                                            '/kaggle/input/qwen2.5/transformers/1.5b/1/config.json']
os.environ.pop('NTL_TF_DIR', None)
print('kaggle lookup:', [lp.find_transformer(n) for n, _, _ in lp.FINE_TUNED], 'qwen:', lp.find_model_dir())
