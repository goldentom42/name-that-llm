"""Name That LLM: Qwen3-4B-Base LoRA classifier alone, averaged over its 5 fold adapters; writes submission.parquet.

Each adapter was trained on the open pool, then continued on 4/5 of train.csv (x5) mixed with 20k pool rows, grouped by
prompt (scripts/run_qwen3_4b_folds.sh; scores in logs/cv_scores.csv). Out-of-fold scores measure one fold model each, so
the value of averaging them is not part of the CV number. No blend and no prompt grouping: this submission tests the model.

Kaggle: GPU T4 x2, internet off. Inputs: the competition data, the Qwen3-4B-Base weights (a directory whose path contains
"4b" and "base", with a config.json of model_type qwen3) and a dataset holding the folders
Qwen3-4B-Base-lora-mix-5fold-fold0/ ... -fold4/. The commit run only sees the 3-row test, so timings come from the log of
this script's hidden run, which Kaggle does not show: BUDGET_H stops starting new folds in time and averages those done.

Folds run in pairs, one per GPU. Models are loaded in the main thread only (from_pretrained changes process-wide torch
state) and the threads only run inference. fp16 because the T4 has no native bf16; rows that overflow are redone in bf16.

Env overrides: NTL_DATA_DIR, NTL_MODEL_DIR, NTL_TF_DIR, NTL_OUT, NTL_DEVICES (comma-separated), NTL_BATCH.
"""
import gc, glob, json, os, threading, time
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ON_KAGGLE = os.path.isdir('/kaggle/input')
ADAPTER = 'Qwen3-4B-Base-lora-mix-5fold-fold{}'
FOLDS = [0, 1, 2, 3, 4]
MAXLEN, BUDGET_H = 384, 8.0  # training's max length; the notebook limit is 9 h


def input_configs():
    return [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/config.json', recursive=True)]


def find_data_dir():
    if os.environ.get('NTL_DATA_DIR'): return os.environ['NTL_DATA_DIR']
    if not ON_KAGGLE: return '.'
    hits = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/train.csv', recursive=True)
            if os.path.exists(os.path.join(os.path.dirname(p), 'test.csv'))]
    assert hits, 'competition data not found under /kaggle/input'
    return hits[0]


def find_base():
    if os.environ.get('NTL_MODEL_DIR'): return os.environ['NTL_MODEL_DIR']
    if not ON_KAGGLE: return 'Qwen/Qwen3-4B-Base'
    flat = lambda c: c.lower().replace('-', '').replace('_', '')
    cfgs = input_configs()
    hits = [c for c in cfgs if '4b' in flat(c) and 'base' in flat(c) and json.load(open(f'{c}/config.json')).get('model_type') == 'qwen3']
    assert hits, f'Qwen3-4B-Base not found under /kaggle/input (config.json dirs: {cfgs})'
    return sorted(hits)[0]


def find_adapter(name):
    if os.environ.get('NTL_TF_DIR'): return os.path.join(os.environ['NTL_TF_DIR'], name)
    if not ON_KAGGLE: return f'models/{name}'
    found = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/adapter_config.json', recursive=True)]
    hits = [d for d in found if os.path.basename(d) == name]
    assert hits, f'attach the dataset holding the {name} adapter as a notebook input (adapter_config.json dirs: {found})'
    return hits[0]


def pick_devices():
    if os.environ.get('NTL_DEVICES'): return os.environ['NTL_DEVICES'].split(',')
    return [f'cuda:{i}' for i in range(torch.cuda.device_count())] or ['cpu']


def load_merged(base_dir, adapter_dir, n_labels, pad_id, device):
    """The base with the LoRA deltas (alpha / r * B @ A, in fp32) merged in by hand and the trained score head copied in."""
    cfg = json.load(open(os.path.join(adapter_dir, 'adapter_config.json')))
    scale = cfg['lora_alpha'] / cfg['r']
    key = 'dtype' if int(transformers.__version__.split('.')[0]) >= 5 else 'torch_dtype'
    dtype = torch.float32 if device == 'cpu' else torch.float16
    m = AutoModelForSequenceClassification.from_pretrained(base_dir, num_labels=n_labels, **{key: dtype}).to(device).eval()
    m.config.pad_token_id = pad_id
    sd = load_file(os.path.join(adapter_dir, 'adapter_model.safetensors'), device=device)
    params, pre, merged = dict(m.named_parameters()), 'base_model.model.', 0
    with torch.no_grad():
        params['score.weight'].copy_(sd[pre + 'score.weight'])
        for k in sd:
            if k.endswith('.lora_A.weight'):
                W = params[k[len(pre):-len('.lora_A.weight')] + '.weight']
                delta = scale * (sd[k.replace('lora_A', 'lora_B')].float() @ sd[k].float())
                W.copy_((W.float() + delta).to(W.dtype)); merged += 1
    expected = m.config.num_hidden_layers * len(cfg['target_modules'])
    assert merged == expected, f'merged {merged} LoRA matrices, expected {expected}'
    return m


@torch.no_grad()
def fill(m, ids, rows, bs, pad_id, device, out):
    """Softmax for `rows` (right padding, last-token pooling as in training), written to out."""
    for s in range(0, len(rows), bs):
        idx = rows[s:s + bs]; L = max(len(ids[i]) for i in idx)
        x = torch.tensor([ids[i] + [pad_id] * (L - len(ids[i])) for i in idx], device=device)
        mask = torch.tensor([[1] * len(ids[i]) + [0] * (L - len(ids[i])) for i in idx], device=device)
        out[idx] = m(input_ids=x, attention_mask=mask).logits.float().softmax(-1).cpu().numpy()


def predict(m, ids, n_labels, bs, pad_id, device, name):
    order = np.argsort([len(x) for x in ids])[::-1]  # longest first: an out-of-memory error shows up at once
    out = np.full((len(ids), n_labels), np.nan, np.float32)
    fill(m, ids, order, bs, pad_id, device, out)
    bad = np.where(~np.isfinite(out).all(1))[0]
    if len(bad) and device != 'cpu':
        print(f'{name}: {len(bad)} rows overflowed in fp16, redone in bf16', flush=True)
        m.bfloat16(); fill(m, ids, bad, min(bs, 8), pad_id, device, out)
    assert np.isfinite(out).all(), f'non-finite probabilities from {name}'
    return out


def main():
    data_dir, base_dir, devices = find_data_dir(), find_base(), pick_devices()
    out_path = os.environ.get('NTL_OUT', '/kaggle/working/submission.parquet' if ON_KAGGLE else 'submission.parquet')
    bs = int(os.environ.get('NTL_BATCH', 32 if devices[0] != 'cpu' else 2))
    print(f'data={data_dir} base={base_dir} devices={devices} batch={bs} out={out_path}', flush=True)
    train, test = pd.read_csv(f'{data_dir}/train.csv', usecols=['model']), pd.read_csv(f'{data_dir}/test.csv')
    labels = sorted(train.model.unique()); assert len(labels) == 18
    tok = AutoTokenizer.from_pretrained(base_dir)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = tok(list(test.text), truncation=True, max_length=MAXLEN)['input_ids']
    print(f'test {len(test)}, mean {np.mean([len(x) for x in ids]):.0f} tokens', flush=True)

    probs, errors, T, last_round = {}, [], time.time(), 0.0
    for r in range(0, len(FOLDS), len(devices)):
        if time.time() - T + last_round > BUDGET_H * 3600:
            print(f'time budget: stopping before folds {FOLDS[r:]}', flush=True); break
        t0 = time.time(); round_folds = FOLDS[r:r + len(devices)]
        models = [load_merged(base_dir, find_adapter(ADAPTER.format(k)), len(labels), pad_id, d) for k, d in zip(round_folds, devices)]

        def work(k, m, d):
            try:
                probs[k] = predict(m, ids, len(labels), bs, pad_id, d, f'fold {k}')
            except Exception as e:  # re-raised in the main thread
                errors.append(e)

        threads = [threading.Thread(target=work, args=a) for a in zip(round_folds, models, devices)]
        for th in threads: th.start()
        for th in threads: th.join()
        if errors: raise errors[0]
        del models; gc.collect()
        if devices[0] != 'cpu': torch.cuda.empty_cache()
        last_round = time.time() - t0
        print(f'folds {round_folds} done in {last_round / 60:.1f} min, {(time.time() - T) / 60:.1f} min so far', flush=True)

    assert probs, 'no fold finished'
    P = np.mean([probs[k] for k in sorted(probs)], 0)
    pred = P.argmax(1)
    print(f'averaged folds {sorted(probs)}; predictions per class {np.bincount(pred, minlength=len(labels)).tolist()}', flush=True)
    sub = pd.DataFrame({'id': test.id.astype(str), 'model': [labels[i] for i in pred]})
    sub.to_parquet(out_path, index=False)
    np.save(out_path.replace('.parquet', '_probs.npy'), P)
    print(sub.head(10).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
