"""Second-stage fine-tuning of a pool-trained checkpoint on train.csv (in-domain data).

Out-of-fold estimate (5 folds grouped by prompt; every fold restarts from the pool checkpoint):
    python src/finetune_train.py models/ModernBERT-base --lr 1e-5 --epochs 4
    -> oof_ft_<name>_lr<lr>_ep<k>.npy for k = 1..epochs, aligned to train.csv rows
Final model on all of train.csv, same schedule, saved after epoch k:
    python src/finetune_train.py models/ModernBERT-base --lr 1e-5 --epochs 4 --final 2
    -> models/<name>-ft/

The schedule always spans --epochs, so the epoch-k snapshot of the final run matches the epoch-k
out-of-fold predictions. Choose k on the selection prompts; the holdout (split_prompts) is report-only.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast, get_cosine_schedule_with_warmup
sys.path.insert(0, os.path.dirname(__file__))
from train_pool import split_prompts


def load(ckpt):
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(ckpt, 'tokenizer.json'), pad_token='[PAD]')
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, dtype=torch.float32).cuda()
    return tok, model


def run(ckpt, ids, y, tr, te, a, on_epoch):
    """Fine-tune on rows tr; after each epoch call on_epoch(epoch, model probs for rows te, model)."""
    torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    tok, model = load(ckpt)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, fused=True)
    steps = a.epochs * ((len(tr) + a.bs - 1) // a.bs)
    sch = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    pad = lambda idx: tok.pad({'input_ids': [ids[i] for i in idx]}, return_tensors='pt').to('cuda')
    for ep in range(1, a.epochs + 1):
        model.train(); perm = rng.permutation(tr)
        for s in range(0, len(perm), a.bs):
            idx = perm[s:s + a.bs]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = model(**pad(idx)).logits
            loss = F.cross_entropy(lg.float(), torch.tensor(y[idx], device='cuda'))
            if not torch.isfinite(loss): raise RuntimeError('non-finite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        P = None
        if len(te):
            model.eval(); P = np.zeros((len(te), y.max() + 1), np.float32)
            order = np.argsort([len(ids[i]) for i in te])
            with torch.no_grad():
                for s in range(0, len(order), 64):
                    j = order[s:s + 64]
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        P[j] = model(**pad(te[j])).logits.float().softmax(-1).cpu().numpy()
        if on_epoch(ep, P, model, tok): break
    del model, opt; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--maxlen', type=int, default=384)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--final', type=int, default=0, help='train on all rows, save after this epoch')
    a = ap.parse_args()

    name = os.path.basename(a.ckpt.rstrip('/'))
    t = pd.read_csv('train.csv'); labels = sorted(t.model.unique())
    y = t.model.map({l: i for i, l in enumerate(labels)}).values
    tok, _ = PreTrainedTokenizerFast(tokenizer_file=os.path.join(a.ckpt, 'tokenizer.json'), pad_token='[PAD]'), None
    ids = tok(t.text.tolist(), truncation=True, max_length=a.maxlen)['input_ids']
    held = split_prompts(t.prompt_id.values)
    T = time.time()

    if a.final:
        out = f'models/{name}-ft'
        def save(ep, P, model, tok_):
            if ep == a.final:
                model.save_pretrained(out); tok_.save_pretrained(out)
                print(f'saved {out} after epoch {ep} ({time.time() - T:.0f}s)', flush=True); return True
        run(a.ckpt, ids, y, np.arange(len(t)), np.array([], int), a, save)
        return

    oof = np.zeros((a.epochs, len(t), len(labels)), np.float32)
    for f, (tr, te) in enumerate(GroupKFold(5).split(t, y, t.prompt_id)):
        def keep(ep, P, model, tok_): oof[ep - 1, te] = P
        run(a.ckpt, ids, y, tr, te, a, keep)
        acc = (oof[a.epochs - 1, te].argmax(1) == y[te]).mean()
        print(f'fold {f} last-epoch acc {acc:.4f} ({time.time() - T:.0f}s)', flush=True)
    base = np.load(f'val_{name}.npy'); ok0 = base.argmax(1) == y
    print(f'pool-only {name}: select {ok0[~held].mean():.4f} holdout {ok0[held].mean():.4f} all {ok0.mean():.4f}')
    tag = f'{name}_lr{a.lr:g}'
    with open('logs/finetune_train.jsonl', 'a') as log:
        for ep in range(1, a.epochs + 1):
            np.save(f'oof_ft_{tag}_ep{ep}.npy', oof[ep - 1])
            ok = oof[ep - 1].argmax(1) == y
            r = dict(model=name, lr=a.lr, ep=ep, select=round(ok[~held].mean(), 4), holdout=round(ok[held].mean(), 4), all=round(ok.mean(), 4))
            print(json.dumps(r), flush=True); log.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
