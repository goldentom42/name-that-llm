import sys; sys.path.insert(0, 'src')
import torch; torch.set_num_threads(2)
import train_qwen_lora as q
tok, m = q.build('Qwen/Qwen2.5-0.5B', [f'l{i}' for i in range(18)], 8, 'cpu')
m = m.float().eval()
texts = ['Short text.', 'A somewhat longer text — with curly ’quotes’… and more words here.']
b = tok(texts, return_tensors='pt', padding=True)
with torch.no_grad():
    batch = m(**b).logits
    single = torch.cat([m(**tok([t], return_tensors='pt')).logits for t in texts])
    # the representation the head should see: hidden state at the last real token of the short text
print('logit scale', float(single.abs().max()))
print('fp32 padded vs unpadded max diff', float((batch - single).abs().max()))
print('attention mask', b.attention_mask.tolist())
