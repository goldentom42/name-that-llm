"""Fine-tune a transformer on the open pool, evaluating on train.csv several times per epoch.

    python src/train_pool.py answerdotai/ModernBERT-base --epochs 3

Checkpoints are selected on 120 train.csv prompts; the other 30 (see split_prompts) are reported
only, so the holdout number is not inflated by selection. Outputs:
    models/<name>/            best checkpoint + tokenizer, loadable offline with from_pretrained
    val_<name>.npy            softmax probs for the best checkpoint, aligned to train.csv rows
    logs/<name>.jsonl         one record per evaluation
"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup


def split_prompts(prompt_ids, n_holdout=30, seed=0):
    """Boolean mask of train.csv rows in the reporting holdout. Shared with blend_pool.py."""
    ids = np.array(sorted(set(prompt_ids)))
    held = np.random.RandomState(seed).choice(ids, n_holdout, replace=False)
    return np.isin(prompt_ids, held)


def sample_weights(pool, y, n_classes, clip):
    """Clipped match_weight, then rebalanced so every class has equal total weight (train.csv is
    balanced at 150 per class, the pool ranges from 3.5k to 22.5k)."""
    w = np.minimum(pool.match_weight.values, clip) if clip > 0 else np.ones(len(pool))
    totals = np.bincount(y, weights=w, minlength=n_classes)
    w = w / totals[y]
    return w / w.mean()


def bucketed_batches(lengths, bs, rng, chunk=256):
    """Shuffle, sort by length inside large chunks, shuffle the batches: less padding, still random."""
    perm = rng.permutation(len(lengths)); out = []
    for s in range(0, len(perm), bs * chunk):
        c = perm[s:s + bs * chunk]; c = c[np.argsort(lengths[c], kind='stable')]
        out += [c[i:i + bs] for i in range(0, len(c), bs)]
    return [out[i] for i in rng.permutation(len(out))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--maxlen', type=int, default=512)
    ap.add_argument('--clip', type=float, default=5.0, help='match_weight cap; 0 = ignore match_weight')
    ap.add_argument('--evals', type=int, default=4, help='evaluations per epoch')
    ap.add_argument('--limit', type=int, default=0, help='pool rows to use (smoke tests)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    name = a.model.split('/')[-1] + a.tag
    os.makedirs('logs', exist_ok=True); out_dir = f'models/{name}'
    torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)

    train = pd.read_csv('train.csv')
    pool = pd.read_parquet('data/open_pool/ntl_open_pool.parquet', columns=['text', 'model', 'match_weight'])
    if a.limit: pool = pool.sample(a.limit, random_state=a.seed).reset_index(drop=True)
    labels = sorted(train.model.unique()); l2i = {l: i for i, l in enumerate(labels)}
    y_pool, y_val = pool.model.map(l2i).values, train.model.map(l2i).values
    assert not np.isnan(y_pool.astype(float)).any(), 'pool label not in train.csv'
    w = sample_weights(pool, y_pool, len(labels), a.clip)
    held = split_prompts(train.prompt_id.values)
    print(f'pool {len(pool)} val {len(train)} (select {(~held).sum()}, holdout {held.sum()}) '
          f'weight max {w.max():.1f} ESS {w.sum()**2 / (w**2).sum():.0f}', flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    enc = lambda s: tok(list(s), truncation=True, max_length=a.maxlen)['input_ids']
    t0 = time.time(); ids_pool, ids_val = enc(pool.text), enc(train.text)
    len_pool = np.array([len(x) for x in ids_pool]); len_val = np.array([len(x) for x in ids_val])
    print(f'tokenised {time.time() - t0:.0f}s, mean {len_pool.mean():.0f} max {len_pool.max()}', flush=True)

    # float32 explicitly: transformers 5.x otherwise loads DeBERTa-v3 as fp16, which gave NaN loss.
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=len(labels), dtype=torch.float32,
        id2label=dict(enumerate(labels)), label2id=l2i).cuda()
    no_decay = ('bias', 'LayerNorm', 'layer_norm', 'norm')
    groups = [{'params': [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], 'weight_decay': 0.01},
              {'params': [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], 'weight_decay': 0.0}]
    opt = torch.optim.AdamW(groups, lr=a.lr, fused=True)
    steps_ep = (len(pool) + a.bs - 1) // a.bs; steps = steps_ep * a.epochs
    sch = get_cosine_schedule_with_warmup(opt, steps // 20, steps)

    def collate(ids_all, idx):
        return tok.pad({'input_ids': [ids_all[i] for i in idx]}, return_tensors='pt').to('cuda')

    @torch.no_grad()
    def evaluate():
        model.eval(); out = np.zeros((len(train), len(labels)), np.float32)
        order = np.argsort(len_val)
        for s in range(0, len(order), 128):
            idx = order[s:s + 128]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = model(**collate(ids_val, idx)).logits
            out[idx] = lg.float().softmax(-1).cpu().numpy()
        model.train(); return out

    def fam(a_): return np.array([labels[i].split('-')[0] for i in a_])

    best, log, step, run_loss, run_n, T = -1.0, open(f'logs/{name}.jsonl', 'a'), 0, 0.0, 0, time.time()
    eval_every = max(1, steps_ep // a.evals)
    for ep in range(a.epochs):
        for idx in bucketed_batches(len_pool, a.bs, rng):
            b = collate(ids_pool, idx)
            yb = torch.tensor(y_pool[idx], device='cuda'); wb = torch.tensor(w[idx], device='cuda', dtype=torch.float32)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = model(**b).logits
            loss = (F.cross_entropy(lg.float(), yb, reduction='none') * wb).mean()
            if not torch.isfinite(loss): raise RuntimeError(f'non-finite loss at step {step}')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            step += 1; run_loss += loss.item(); run_n += 1
            if step % eval_every == 0 or step == steps:
                P = evaluate(); pred = P.argmax(1)
                ok = pred == y_val
                r = dict(step=step, ep=round(step / steps_ep, 2), loss=round(run_loss / run_n, 4),
                         select=round(ok[~held].mean(), 4), holdout=round(ok[held].mean(), 4), all=round(ok.mean(), 4),
                         family=round((fam(pred) == fam(y_val)).mean(), 4), min_class_preds=int(np.bincount(pred, minlength=len(labels)).min()),
                         max_class_preds=int(np.bincount(pred, minlength=len(labels)).max()), mins=round((time.time() - T) / 60, 1))
                saved = r['select'] > best
                if saved:
                    best = r['select']; model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
                    np.save(f'val_{name}.npy', P)
                print(json.dumps(r) + (' *' if saved else ''), flush=True); log.write(json.dumps(r) + '\n'); log.flush()
                run_loss, run_n = 0.0, 0
    print(f'done, best select {best:.4f}, {(time.time() - T) / 60:.0f} min', flush=True)


if __name__ == '__main__':
    main()
