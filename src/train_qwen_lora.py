"""LoRA fine-tune of a Qwen2.5 decoder as an 18-way classifier (last-token pooling) on the open pool.

Pilot:  python src/train_qwen_lora.py Qwen/Qwen2.5-1.5B --rows 50000 --epochs 1
Evaluates on train.csv several times per epoch; checkpoint selection uses the 120 selection prompts,
the 30-prompt holdout is report-only (split_prompts in train_pool.py). Outputs:
    models/<name>/           best LoRA adapter + trained score head (peft save_pretrained)
    val_<name>.npy           softmax probs of the best evaluation, aligned to train.csv rows
    logs/<name>.jsonl        one record per evaluation
With --ckpt_every N, the trainable weights, optimizer, scheduler and RNG states are saved to
models/<name>-resume.pt every N steps; rerun the same command with --resume to continue after a crash
(batches are regenerated from the seed and the done ones skipped). The file is deleted when the run ends.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
sys.path.insert(0, os.path.dirname(__file__))
from train_pool import bucketed_batches, sample_weights, split_prompts


def build(base, labels, r, device):
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(labels), dtype=torch.bfloat16,
        id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
    model.config.pad_token_id = tok.pad_token_id
    # Without this, a 16 x 384-token batch needs more than 24 GB of activations (28 layers, 8960-wide MLP).
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, task_type='SEQ_CLS',
                     target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                     modules_to_save=['score'])
    model = get_peft_model(model, cfg)
    for p in model.parameters():  # frozen base stays bf16; trainable LoRA + head in fp32 for AdamW
        if p.requires_grad: p.data = p.data.float()
    return tok, model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base')
    ap.add_argument('--rows', type=int, default=50000, help='pool rows to sample (0 = all)')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--r', type=int, default=16)
    ap.add_argument('--maxlen', type=int, default=384)
    ap.add_argument('--clip', type=float, default=5.0)
    ap.add_argument('--evals', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--ckpt_every', type=int, default=0, help='save a resume checkpoint every N steps (0 = never)')
    ap.add_argument('--resume', action='store_true', help='continue from models/<name>-resume.pt if it exists')
    a = ap.parse_args()

    name = a.base.split('/')[-1] + f'-lora{a.tag}'
    os.makedirs('logs', exist_ok=True); out_dir = f'models/{name}'
    torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    train = pd.read_csv('train.csv'); labels = sorted(train.model.unique()); l2i = {l: i for i, l in enumerate(labels)}
    pool = pd.read_parquet('data/open_pool/ntl_open_pool.parquet', columns=['text', 'model', 'match_weight'])
    if a.rows: pool = pool.sample(a.rows, random_state=a.seed).reset_index(drop=True)
    y_pool, y_val = pool.model.map(l2i).values, train.model.map(l2i).values
    w = sample_weights(pool, y_pool, len(labels), a.clip); held = split_prompts(train.prompt_id.values)

    tok, model = build(a.base, labels, a.r, 'cuda')
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc = lambda s: tok(list(s), truncation=True, max_length=a.maxlen)['input_ids']
    ids_pool, ids_val = enc(pool.text), enc(train.text)
    len_pool, len_val = np.array([len(x) for x in ids_pool]), np.array([len(x) for x in ids_val])
    print(f'pool {len(pool)} trainable {n_train / 1e6:.1f}M tokens mean {len_pool.mean():.0f} max {len_pool.max()}', flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01, fused=True)
    steps_ep = (len(pool) + a.bs - 1) // a.bs; steps = steps_ep * a.epochs
    sch = get_cosine_schedule_with_warmup(opt, steps // 20, steps)
    pad = lambda ids_all, idx: tok.pad({'input_ids': [ids_all[i] for i in idx]}, return_tensors='pt').to('cuda')

    def step(idx):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg = model(**pad(ids_pool, idx)).logits
        loss = (F.cross_entropy(lg.float(), torch.tensor(y_pool[idx], device='cuda'), reduction='none')
                * torch.tensor(w[idx], device='cuda', dtype=torch.float32)).mean()
        if not torch.isfinite(loss): raise RuntimeError('non-finite loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        return loss.item()

    @torch.no_grad()
    def evaluate():
        model.eval(); out = np.zeros((len(train), len(labels)), np.float32); order = np.argsort(len_val)
        for s in range(0, len(order), 64):
            idx = order[s:s + 64]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out[idx] = model(**pad(ids_val, idx)).logits.float().softmax(-1).cpu().numpy()
        model.train(); return out

    # Memory probe: the longest batch first, so an OOM shows up in seconds rather than mid-run.
    torch.cuda.reset_peak_memory_stats(); model.train()
    step(np.argsort(len_pool)[-a.bs:]); opt.zero_grad(set_to_none=True)
    print(f'longest batch ok, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB', flush=True)

    trainable = {k: p for k, p in model.named_parameters() if p.requires_grad}
    resume_path, start, best = f'models/{name}-resume.pt', 0, -1.0
    if a.resume and os.path.exists(resume_path):
        st = torch.load(resume_path, map_location='cuda', weights_only=False)
        for k, v in st['params'].items(): trainable[k].data.copy_(v)
        opt.load_state_dict(st['opt']); sch.load_state_dict(st['sch'])
        torch.set_rng_state(st['cpu_rng'].cpu()); torch.cuda.set_rng_state(st['cuda_rng'].cpu())
        start, best = st['n'], st['best']
        print(f'resumed from step {start} of {steps}, best select so far {best:.4f}', flush=True)
    elif a.resume:
        print(f'--resume: no {resume_path}, starting from scratch', flush=True)

    def save_resume(n):
        st = dict(params={k: p.detach().cpu() for k, p in trainable.items()}, opt=opt.state_dict(), sch=sch.state_dict(),
                  cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), n=n, best=best)
        torch.save(st, resume_path + '.tmp'); os.replace(resume_path + '.tmp', resume_path)  # never a half-written file

    log, n, run_loss, T = open(f'logs/{name}.jsonl', 'a'), 0, [], time.time()
    eval_every = max(1, steps_ep // a.evals)
    for ep in range(a.epochs):
        for idx in bucketed_batches(len_pool, a.bs, rng):  # same seed -> same batches, so a resumed run skips the done ones
            n += 1
            if n <= start: continue
            run_loss.append(step(idx))
            if n == start + 200: print(f'speed {200 * a.bs / (time.time() - T):.1f} rows/s -> {(steps - start) * a.bs / (200 * a.bs / (time.time() - T)) / 60:.0f} min for this run', flush=True)
            if n % eval_every == 0 or n == steps:
                P = evaluate(); ok = P.argmax(1) == y_val; pred = P.argmax(1)
                r = dict(step=n, ep=round(n / steps_ep, 2), loss=round(float(np.mean(run_loss)), 4),
                         select=round(ok[~held].mean(), 4), holdout=round(ok[held].mean(), 4), all=round(ok.mean(), 4),
                         min_class_preds=int(np.bincount(pred, minlength=len(labels)).min()),
                         max_class_preds=int(np.bincount(pred, minlength=len(labels)).max()), mins=round((time.time() - T) / 60, 1))
                saved = r['select'] > best
                if saved:
                    best = r['select']; model.save_pretrained(out_dir); tok.save_pretrained(out_dir); np.save(f'val_{name}.npy', P)
                print(json.dumps(r) + (' *' if saved else ''), flush=True); log.write(json.dumps(r) + '\n'); log.flush(); os.fsync(log.fileno())
                run_loss = []
            if a.ckpt_every and n % a.ckpt_every == 0 and n < steps:
                save_resume(n); print(f'resume checkpoint at step {n}, {(time.time() - T) / 60:.0f} min', flush=True)
    if os.path.exists(resume_path): os.remove(resume_path)
    print(f'done, best select {best:.4f}, {(time.time() - T) / 60:.0f} min', flush=True)


if __name__ == '__main__':
    main()
