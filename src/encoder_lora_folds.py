"""LoRA fine-tuning of a pool-trained encoder (default ModernBERT) on train.csv folds, and does averaging the folds help?

Starts from the checkpoint trained on the open pool, adds a LoRA adapter on the attention and MLP projections, and
trains the adapter plus the classification head on train.csv rows. Folds are grouped by prompt. Modes:

  --mode oof   K folds on all prompts. Out-of-fold accuracy per epoch, comparable with val_<name>.npy (pool only) and
               with full fine-tuning. Writes oof_lora_<name><tag>.npy. With --save, each fold's adapter is merged into
               a plain checkpoint, models/<name>-lora<tag>-fold<k>/, which the Kaggle script loads like any encoder.
  --mode avg   Out-of-fold accuracy cannot show the value of averaging (each row is predicted by one model only), so:
               hold out prompts (--outer folds), and on the rest train K fold models and one model on all of the rest.
               On the held-out prompts compare the average of the K models with the single model.

Results are appended to logs/encoder_lora.jsonl.

    python src/encoder_lora_folds.py models/ModernBERT-base --mode oof --lr 1e-4 --epochs 6
    python src/encoder_lora_folds.py models/ModernBERT-base --mode avg --lr 1e-4 --epochs 6
"""
import argparse, json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')  # the machine resets under many-thread load
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast, get_cosine_schedule_with_warmup
sys.path.insert(0, os.path.dirname(__file__))
from train_pool import bucketed_batches, split_prompts

TARGETS = r'model\.layers\.\d+\.(attn\.(Wqkv|Wo)|mlp\.(Wi|Wo))'


class Trainer:
    def __init__(self, a, labels, ids, y, tok):
        self.a, self.labels, self.ids, self.y, self.tok = a, labels, ids, y, tok
        self.lens = np.array([len(x) for x in ids])

    def pad(self, rows):
        return self.tok.pad({'input_ids': [self.ids[i] for i in rows]}, return_tensors='pt').to(self.a.device)

    def build(self):
        a = self.a
        m = AutoModelForSequenceClassification.from_pretrained(a.ckpt, dtype=torch.float32)
        assert [m.config.id2label[i] for i in range(len(self.labels))] == self.labels
        cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, target_modules=TARGETS, modules_to_save=['head', 'classifier'])
        return get_peft_model(m, cfg).to(a.device)

    @torch.no_grad()
    def predict(self, model, rows):
        model.eval(); out = np.zeros((len(rows), len(self.labels)), np.float32); order = np.argsort(self.lens[rows])
        for s in range(0, len(order), 64):
            j = order[s:s + 64]
            with torch.autocast(self.a.device, dtype=torch.bfloat16):
                out[j] = model(**self.pad(rows[j])).logits.float().softmax(-1).cpu().numpy()
        model.train(); return out

    def fit(self, tr, te, seed):
        """Train on rows tr; return ({epoch: probabilities on rows te}, the model)."""
        a = self.a; torch.manual_seed(seed); rng = np.random.RandomState(seed)
        model = self.build(); params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
        steps = a.epochs * ((len(tr) + a.bs - 1) // a.bs)
        sch = get_cosine_schedule_with_warmup(opt, steps // 20, steps)
        model.train(); out = {}
        for ep in range(1, a.epochs + 1):
            for b in bucketed_batches(self.lens[tr], a.bs, rng):
                rows = tr[b]
                with torch.autocast(a.device, dtype=torch.bfloat16):
                    lg = model(**self.pad(rows)).logits
                loss = F.cross_entropy(lg.float(), torch.tensor(self.y[rows], device=a.device))
                if not torch.isfinite(loss): raise RuntimeError('non-finite loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            out[ep] = self.predict(model, te)
        return out, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt', nargs='?', default='models/ModernBERT-base')
    ap.add_argument('--mode', choices=['oof', 'avg'], default='oof')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--outer', type=int, default=3, help='--mode avg: number of held-out prompt groups')
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--r', type=int, default=16)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--maxlen', type=int, default=384, help='384 is what the Kaggle script uses')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--save', action='store_true', help='--mode oof: save each fold as a merged checkpoint')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max_prompts', type=int, default=0, help='use only the first N prompts (smoke test)')
    a = ap.parse_args()

    name = os.path.basename(a.ckpt.rstrip('/'))
    t = pd.read_csv('train.csv'); labels = sorted(t.model.unique()); keep = np.ones(len(t), bool)
    held = split_prompts(t.prompt_id.values); base = np.load(f'val_{name}.npy')
    if a.max_prompts: keep = t.prompt_id.isin(t.prompt_id.unique()[:a.max_prompts]).values
    t, held, base = t[keep].reset_index(drop=True), held[keep], base[keep]
    y = t.model.map({l: i for i, l in enumerate(labels)}).values; groups = t.prompt_id.values
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(a.ckpt, 'tokenizer.json'), pad_token='[PAD]')
    ids = tok(list(t.text), truncation=True, max_length=a.maxlen)['input_ids']
    tr = Trainer(a, labels, ids, y, tok); T = time.time()
    settings = {k: v for k, v in vars(a).items() if k not in ('ckpt', 'device', 'save')}
    log = open('logs/encoder_lora.jsonl', 'a')

    def emit(**r):
        r = dict(model=name, **r, **settings); print(json.dumps(r), flush=True)
        log.write(json.dumps(r) + '\n'); log.flush(); os.fsync(log.fileno())

    ok = lambda P, rows: P.argmax(1) == y[rows]
    if a.mode == 'oof':
        oof = {ep: np.zeros((len(t), len(labels)), np.float32) for ep in range(1, a.epochs + 1)}
        for f, (tri, te) in enumerate(GroupKFold(a.folds).split(t, y, groups)):
            P, model = tr.fit(tri, te, a.seed + f)
            for ep in P: oof[ep][te] = P[ep]
            emit(kind='fold', fold=f, acc=round(float(ok(P[a.epochs], te).mean()), 4), pool_only=round(float((base[te].argmax(1) == y[te]).mean()), 4),
                 mins=round((time.time() - T) / 60, 1))
            if a.save:
                out = f'models/{name}-lora{a.tag}-fold{f}'; model.merge_and_unload().save_pretrained(out); tok.save_pretrained(out)
            del model; torch.cuda.empty_cache()
        for ep in oof:
            c = oof[ep].argmax(1) == y
            emit(kind='oof', epoch=ep, select=round(float(c[~held].mean()), 4), holdout=round(float(c[held].mean()), 4), all=round(float(c.mean()), 4))
        np.save(f'oof_lora_{name}{a.tag}.npy', oof[a.epochs])
        c0 = base.argmax(1) == y
        print(f'pool only: select {c0[~held].mean():.4f} holdout {c0[held].mean():.4f} all {c0.mean():.4f}; {(time.time() - T) / 60:.0f} min', flush=True)
    else:
        n_correct = dict(base=0, single=0, fold_mean=0, avg=0); n = 0; rows = np.arange(len(t))
        for o, (rest, held_out) in enumerate(GroupKFold(a.outer).split(rows, y, groups)):
            single = tr.fit(rest, held_out, a.seed + 100 * o)[0][a.epochs]; ps = []
            for f, (i, _) in enumerate(GroupKFold(a.folds).split(rest, y[rest], groups[rest])):
                ps.append(tr.fit(rest[i], held_out, a.seed + 100 * o + f + 1)[0][a.epochs]); torch.cuda.empty_cache()
            r = dict(base=ok(base[held_out], held_out), single=ok(single, held_out), avg=ok(np.mean(ps, 0), held_out),
                     fold_mean=np.mean([ok(p, held_out) for p in ps], 0))
            for k in r: n_correct[k] += float(r[k].sum())
            n += len(held_out)
            emit(kind='outer', outer_fold=o, **{k: round(float(v.mean()), 4) for k, v in r.items()}, mins=round((time.time() - T) / 60, 1))
        emit(kind='avg_summary', **{k: round(v / n, 4) for k, v in n_correct.items()}, avg_minus_single=round((n_correct['avg'] - n_correct['single']) / n, 4))
    print(f'done, {(time.time() - T) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
