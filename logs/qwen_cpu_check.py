"""CPU sanity check of the LoRA classifier wiring with Qwen2.5-0.5B, 2 threads."""
import sys; sys.path.insert(0, 'src')
import torch; torch.set_num_threads(2)
import train_qwen_lora as q
labels = [f'l{i}' for i in range(18)]
tok, m = q.build('Qwen/Qwen2.5-0.5B', labels, 8, 'cpu')
m.print_trainable_parameters()
b = tok(['Short text.', 'A somewhat longer text — with curly ’quotes’…'], return_tensors='pt', padding=True)
ac = lambda: torch.autocast('cpu', dtype=torch.bfloat16)
with ac(): out = m(**b)
loss = torch.nn.functional.cross_entropy(out.logits.float(), torch.tensor([1, 2])); loss.backward()
print('logits', tuple(out.logits.shape), 'loss', round(loss.item(), 3), 'pad id', tok.pad_token_id, 'cfg pad', m.config.pad_token_id)
# Last-token pooling must ignore right padding: the short text alone gives the same logits as in the batch.
m.eval()
with torch.no_grad(), ac():
    a1 = m(**tok(['Short text.'], return_tensors='pt')).logits; a2 = m(**b).logits[:1]
print('padding-invariant', torch.allclose(a1.float(), a2.float(), atol=2e-2), float((a1 - a2).abs().max()))
grads = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is not None]
print('trainable with grads', len(grads), 'score head trained', any('score' in n for n in grads), 'dtypes', {str(p.dtype) for p in m.parameters() if p.requires_grad})
