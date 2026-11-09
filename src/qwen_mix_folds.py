"""Does in-domain data help the Qwen LoRA classifier? 2-fold test on train.csv, grouped by prompt.

For each fold: continue training the pool-trained adapter (--adapter, default models/Qwen2.5-1.5B-lora-full) on
the other fold's train.csv rows (repeated --repeat times) mixed with --pool_rows fresh pool rows,
then predict the held fold. Together the folds give out-of-fold predictions for every train.csv
row, comparable with val_<adapter name>.npy (pool only). Evaluated at 50% and 100% of
the schedule. Outputs oof_qwen_mix<tag>_half.npy / _full.npy (oof_mix_<base name><tag>_* for another
--base) and logs/qwen_mix.jsonl.
With --final: one run on all train.csv rows (same recipe), adapter saved to models/<base name>-lora-mix<tag>/.

    python src/qwen_mix_folds.py --repeat 5 --pool_rows 20000 --lr 3e-5
    python src/qwen_mix_folds.py --base Qwen/Qwen3-4B-Base --adapter models/Qwen3-4B-Base-lora-full
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from peft import PeftModel
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
sys.path.insert(0, os.path.dirname(__file__))
from train_pool import bucketed_batches, sample_weights, split_prompts

DEFAULT_BASE = 'Qwen/Qwen2.5-1.5B'


def load(labels, BASE, ADAPTER):
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    m = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(labels), dtype=torch.bfloat16,
        id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
    m.config.pad_token_id = tok.pad_token_id
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    m = PeftModel.from_pretrained(m, ADAPTER, is_trainable=True)
    for p in m.parameters():
        if p.requires_grad: p.data = p.data.float()
    return tok, m.cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=DEFAULT_BASE)
    ap.add_argument('--adapter', default='models/Qwen2.5-1.5B-lora-full', help='pool-trained LoRA adapter to continue')
    ap.add_argument('--repeat', type=int, default=5, help='times each in-fold train.csv row is seen')
    ap.add_argument('--pool_rows', type=int, default=20000)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--maxlen', type=int, default=384)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--final', action='store_true', help='train once on all train.csv rows + pool, save the adapter')
    a = ap.parse_args()

    t = pd.read_csv('train.csv'); labels = sorted(t.model.unique()); l2i = {l: i for i, l in enumerate(labels)}
    y_t = t.model.map(l2i).values; held = split_prompts(t.prompt_id.values)
    pool = pd.read_parquet('data/open_pool/ntl_open_pool.parquet', columns=['text', 'model', 'match_weight'])
    # A fresh random sample (every pool row was already seen once in the first epoch).
    pool = pool.sample(a.pool_rows, random_state=a.seed + 1).reset_index(drop=True)
    y_p = pool.model.map(l2i).values; w_p = sample_weights(pool, y_p, len(labels), 5.0)
    base_name = a.base.rstrip('/').split('/')[-1]
    oof_prefix = 'oof_qwen_mix' if a.base == DEFAULT_BASE else f'oof_mix_{base_name}'  # keep the 1.5B file names
    base = np.load(f"val_{os.path.basename(a.adapter.rstrip('/'))}.npy")
    oof = {'half': np.zeros((len(t), len(labels)), np.float32), 'full': np.zeros((len(t), len(labels)), np.float32)}
    log = open('logs/qwen_mix.jsonl', 'a'); T = time.time()

    splits = [(np.arange(len(t)), np.array([], int))] if a.final else list(GroupKFold(2).split(t, y_t, t.prompt_id))
    for f, (tr, te) in enumerate(splits):
        torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed + f)
        tok, model = load(labels, a.base, a.adapter)
        enc = lambda s: tok(list(s), truncation=True, max_length=a.maxlen)['input_ids']
        ids_t, ids_p = enc(t.text), enc(pool.text)
        # Mixed training set: pool rows (class-balanced weights, mean 1) + in-fold train rows x repeat (weight 1).
        ids = ids_p + [ids_t[i] for i in np.tile(tr, a.repeat)]
        y = np.concatenate([y_p, np.tile(y_t[tr], a.repeat)])
        w = np.concatenate([w_p, np.ones(len(tr) * a.repeat)]).astype(np.float32)
        lens = np.array([len(x) for x in ids])
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01, fused=True)
        steps = (len(ids) + a.bs - 1) // a.bs
        sch = get_cosine_schedule_with_warmup(opt, steps // 20, steps)
        pad = lambda seqs: tok.pad({'input_ids': seqs}, return_tensors='pt').to('cuda')

        @torch.no_grad()
        def predict():
            model.eval(); out = np.zeros((len(te), len(labels)), np.float32); order = np.argsort([len(ids_t[i]) for i in te])
            for s in range(0, len(order), 64):
                j = order[s:s + 64]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out[j] = model(**pad([ids_t[i] for i in te[j]])).logits.float().softmax(-1).cpu().numpy()
            model.train(); return out

        print(f'fold {f}: train rows {len(tr)} x{a.repeat} + pool {len(pool)} = {len(ids)}, {steps} steps', flush=True)
        model.train(); n = 0; run_loss = []
        for idx in bucketed_batches(lens, a.bs, rng):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = model(**pad([ids[i] for i in idx])).logits
            loss = (F.cross_entropy(lg.float(), torch.tensor(y[idx], device='cuda'), reduction='none')
                    * torch.tensor(w[idx], device='cuda')).mean()
            if not torch.isfinite(loss): raise RuntimeError('non-finite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True); run_loss.append(loss.item()); n += 1
            if not a.final and n in (steps // 2, steps):
                stage = 'half' if n < steps else 'full'; P = predict(); oof[stage][te] = P
                ok, ok0 = P.argmax(1) == y_t[te], base[te].argmax(1) == y_t[te]
                r = dict(fold=f, stage=stage, step=n, loss=round(float(np.mean(run_loss)), 4), acc=round(ok.mean(), 4),
                         pool_only_acc=round(ok0.mean(), 4), mins=round((time.time() - T) / 60, 1), **vars(a))
                print(json.dumps(r), flush=True); log.write(json.dumps(r) + '\n'); log.flush(); os.fsync(log.fileno()); run_loss = []
        if a.final:
            out = f'models/{base_name}-lora-mix{a.tag}'; model.save_pretrained(out)
            print(f'saved {out}: {n} steps, last-500 loss {np.mean(run_loss[-500:]):.4f}, {(time.time() - T) / 60:.0f} min', flush=True); return
        del model, opt; torch.cuda.empty_cache()

    for stage, P in oof.items():
        np.save(f'{oof_prefix}{a.tag}_{stage}.npy', P); ok = P.argmax(1) == y_t
        r = dict(fold='all', stage=stage, select=round(ok[~held].mean(), 4), holdout=round(ok[held].mean(), 4), all=round(ok.mean(), 4), **vars(a))
        print(json.dumps(r), flush=True); log.write(json.dumps(r) + '\n')
    ok0 = base.argmax(1) == y_t
    print(f'pool only: select {ok0[~held].mean():.4f} holdout {ok0[held].mean():.4f} all {ok0.mean():.4f}; {(time.time() - T) / 60:.0f} min', flush=True)


if __name__ == '__main__':
    main()
