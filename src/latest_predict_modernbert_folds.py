"""Name That LLM: ModernBERT-base LoRA classifier alone, averaged over its 5 fold checkpoints; writes submission.parquet.

Each fold started from the pool-trained ModernBERT and trained a LoRA adapter plus the head on 4/5 of train.csv, grouped by
prompt (src/encoder_lora_folds.py --lr 1e-4 --epochs 6 --save; 57.05% 5-fold CV), then was merged into a plain checkpoint.
Out-of-fold scores measure one fold model each, so the value of averaging them is not part of the CV number.
No blend and no prompt grouping: this submission tests the model.

Kaggle: GPU T4, internet off. Inputs: the competition data and a dataset holding the folders
ModernBERT-base-lora-fold0/ ... -fold4/. fp16 on GPU because the T4 has no native bf16.

Env overrides: NTL_DATA_DIR, NTL_TF_DIR, NTL_OUT, NTL_DEVICE, NTL_BATCH.
"""
import gc, glob, os, time
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np, pandas as pd, torch, transformers
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast

ON_KAGGLE = os.path.isdir('/kaggle/input')
CHECKPOINT = 'ModernBERT-base-lora-fold{}'
FOLDS = [0, 1, 2, 3, 4]
MAXLEN = 384  # training's max length


def find_data_dir():
    if os.environ.get('NTL_DATA_DIR'): return os.environ['NTL_DATA_DIR']
    if not ON_KAGGLE: return '.'
    hits = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/train.csv', recursive=True)
            if os.path.exists(os.path.join(os.path.dirname(p), 'test.csv'))]
    assert hits, 'competition data not found under /kaggle/input'
    return hits[0]


def find_checkpoint(name):
    if os.environ.get('NTL_TF_DIR'): return os.path.join(os.environ['NTL_TF_DIR'], name)
    if not ON_KAGGLE: return f'models/{name}'
    found = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/config.json', recursive=True)]
    hits = [d for d in found if os.path.basename(d) == name]
    assert hits, f'attach the dataset holding the {name} checkpoint as a notebook input (config.json dirs: {found})'
    return hits[0]


def pick_device():
    if os.environ.get('NTL_DEVICE'): return os.environ['NTL_DEVICE']
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def fold_probs(model_dir, labels, texts, device, bs):
    """Softmax of one fold checkpoint. The tokenizer is built from tokenizer.json: its config, written by transformers 5.x, does not load on 4.x."""
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, 'tokenizer.json'), pad_token='[PAD]')
    key = 'dtype' if int(transformers.__version__.split('.')[0]) >= 5 else 'torch_dtype'
    dtype = torch.float32 if device == 'cpu' else torch.float16
    m = AutoModelForSequenceClassification.from_pretrained(model_dir, **{key: dtype}).to(device).eval()
    assert [m.config.id2label[i] for i in range(len(labels))] == list(labels), 'label order differs from train.csv'
    ids = tok(list(texts), truncation=True, max_length=MAXLEN)['input_ids']
    order = np.argsort([len(x) for x in ids])[::-1]; out = np.zeros((len(ids), len(labels)), np.float32)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        b = tok.pad({'input_ids': [ids[i] for i in idx]}, return_tensors='pt').to(device)
        out[idx] = m(**b).logits.float().softmax(-1).cpu().numpy()
    assert np.isfinite(out).all(), f'non-finite probabilities from {model_dir}'
    del m; gc.collect()
    if device != 'cpu': torch.cuda.empty_cache()
    return out


def main():
    data_dir, device = find_data_dir(), pick_device()
    out_path = os.environ.get('NTL_OUT', '/kaggle/working/submission.parquet' if ON_KAGGLE else 'submission.parquet')
    bs = int(os.environ.get('NTL_BATCH', 32 if device != 'cpu' else 4))
    print(f'data={data_dir} device={device} batch={bs} out={out_path}', flush=True)
    train, test = pd.read_csv(f'{data_dir}/train.csv', usecols=['model']), pd.read_csv(f'{data_dir}/test.csv')
    labels = sorted(train.model.unique()); assert len(labels) == 18
    T = time.time(); probs = []
    for k in FOLDS:
        probs.append(fold_probs(find_checkpoint(CHECKPOINT.format(k)), labels, test.text, device, bs))
        print(f'fold {k} done, {(time.time() - T) / 60:.1f} min', flush=True)
    P = np.mean(probs, 0); pred = P.argmax(1)
    print(f'averaged {len(probs)} folds; predictions per class {np.bincount(pred, minlength=len(labels)).tolist()}', flush=True)
    sub = pd.DataFrame({'id': test.id.astype(str), 'model': [labels[i] for i in pred]})
    sub.to_parquet(out_path, index=False)
    np.save(out_path.replace('.parquet', '_probs.npy'), P)
    print(sub.head(10).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
