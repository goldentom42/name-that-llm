"""Light end-to-end test of latest_predict.main(): TF-IDF, pool model and Qwen features stubbed; real fine-tuned checkpoints + LoRA on GPU."""
import os, sys
sys.path.insert(0, 'src')
import numpy as np
import latest_predict as lp
rng = np.random.RandomState(0)
def rand_probs(n): P = rng.rand(n, 18); return P / P.sum(1, keepdims=True)
class Stub:
    def __init__(self, *a): pass
    def __call__(self, texts, batch): return np.random.RandomState(len(texts)).normal(size=(len(texts), 8)).astype(np.float32)
lp.tfidf_probs = lambda tr, y, te: rand_probs(len(te))
lp.pool_probs = lambda path, labels, te: rand_probs(len(te))
lp.LLMFeaturizer = Stub
os.environ['NTL_OUT'] = 'logs/stub_test/sub_lora.parquet'
lp.main()
lp.ON_KAGGLE = True
lp.glob.glob = lambda pat, recursive=False: ['/kaggle/input/ntl-qwen-lora/Qwen2.5-1.5B-lora-full/adapter_config.json'] if 'adapter' in pat else []
print('kaggle adapter lookup ->', lp.find_adapter('Qwen2.5-1.5B-lora-full'))
