import sys, time, pandas as pd, numpy as np, torch
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'microsoft/deberta-v3-base'
FOLD, EPOCHS, BS, LR, MAXLEN = 0, int(sys.argv[2]) if len(sys.argv) > 2 else 6, 8, 3e-5, 512
torch.manual_seed(0)
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
d = pd.read_csv('train.csv')
labels = sorted(d.model.unique()); l2i = {l: i for i, l in enumerate(labels)}
y = d.model.map(l2i).values
tr, va = list(GroupKFold(5).split(d, y, d.prompt_id))[FOLD]
tok = AutoTokenizer.from_pretrained(MODEL)
enc = lambda idx: tok(list(d.text.iloc[idx]), truncation=True, max_length=MAXLEN)
etr, eva = enc(tr), enc(va)
print('max tokens', max(len(x) for x in etr['input_ids']), flush=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(labels), dtype=torch.float32).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * ((len(tr) + BS - 1) // BS)
sch = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
def batch(e, ids):
    b = tok.pad({'input_ids': [e['input_ids'][i] for i in ids], 'attention_mask': [e['attention_mask'][i] for i in ids]}, return_tensors='pt')
    return {k: v.to(dev) for k, v in b.items()}
def evaluate():
    model.eval(); P = []
    order = np.argsort([len(x) for x in eva['input_ids']])
    with torch.no_grad():
        for s in range(0, len(va), 32):
            ids = order[s:s+32]; P.append((ids, model(**batch(eva, ids)).logits.float().cpu()))
    out = torch.zeros(len(va), len(labels))
    for ids, lg in P: out[ids] = lg
    model.train(); return out
for ep in range(EPOCHS):
    t = time.time(); perm = np.random.RandomState(ep).permutation(len(tr)); tot = 0
    for s in range(0, len(tr), BS):
        ids = perm[s:s+BS]
        loss = torch.nn.functional.cross_entropy(model(**batch(etr, ids)).logits, torch.tensor(y[tr][ids], device=dev))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(); tot += loss.item()
    lg = evaluate(); acc = (lg.argmax(1).numpy() == y[va]).mean()
    print(f'ep {ep} loss {tot/(len(tr)/BS):.3f} val_acc {acc:.4f} ({time.time()-t:.0f}s)', flush=True)
np.save(f'oof_fold{FOLD}_{MODEL.split("/")[-1]}.npy', lg.numpy())
