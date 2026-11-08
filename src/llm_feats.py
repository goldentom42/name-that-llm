import sys, time, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'Qwen/Qwen2.5-1.5B'
dev = 'mps'
d = pd.read_csv('train.csv')
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
L = m.config.num_hidden_layers; layers = list(range(4, L + 1, 4))
print('layers', L, 'saving', layers, flush=True)
E = {l: [] for l in layers}; PPL = []
t = time.time()
with torch.no_grad():
    for i, txt in enumerate(d.text):
        ids = tok(txt, return_tensors='pt').input_ids.to(dev)
        out = m(ids, output_hidden_states=True)
        for l in layers: E[l].append(out.hidden_states[l][0].float().mean(0).cpu().numpy())
        lp = torch.log_softmax(out.logits[0, :-1].float(), -1).gather(1, ids[0, 1:, None])[:, 0]
        r = (out.logits[0, :-1].float().argsort(-1, descending=True) == ids[0, 1:, None]).float().argmax(-1)
        PPL.append([lp.mean().item(), lp.std().item(), lp.min().item(), np.median(lp.cpu()), (r == 0).float().mean().item(), (r < 5).float().mean().item()])
        if i % 300 == 0: print(i, f'{time.time()-t:.0f}s', 'nan' if np.isnan(E[layers[-1]][-1]).any() else 'ok', flush=True)
name = MODEL.split('/')[-1]
np.savez(f'feats_{name}.npz', ppl=np.array(PPL), **{f'l{l}': np.array(E[l]) for l in layers})
print('done', f'{time.time()-t:.0f}s', flush=True)
