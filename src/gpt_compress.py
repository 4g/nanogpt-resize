import copy

import torch
from nanogpt_model import GPT
from train import sample, train
import tiktoken

prompt = "Indian Prime minister Jawaharlal"

tokenizer = tiktoken.get_encoding('gpt2')
large_model = GPT.from_pretrained('gpt2').cuda()
print(large_model)

s = sample(large_model, tokenizer, prompt=prompt, temperature=0.8, top_k=500, max_tokens=200)
print(s)

small_model_args = copy.deepcopy(large_model.config)
# small_model_args.vocab_size = 50304
small_model_args.n_embd = small_model_args.n_embd // 2
small_model = GPT(config=small_model_args).cuda()

s = sample(small_model, tokenizer, prompt=prompt, temperature=0.8, top_k=500, max_tokens=200)

print("largeparams", sum(p.numel() for p in large_model.parameters()))
print("smallparams", sum(p.numel() for p in small_model.parameters()))

for (large_name, large_p), (small_name, small_p) in zip(large_model.named_parameters(), small_model.named_parameters()):
    small_p.requires_grad = False

    gap = [1] * (4 - len(large_p.shape))
    out_shape = small_p.shape if len(small_p.shape) > 1 else (1, *small_p.shape)
    out = torch.nn.functional.interpolate(large_p.view(*gap, *large_p.shape), size=out_shape, mode='nearest-exact')
    out = out.squeeze()
    assert out.shape == small_p.shape
    small_p.copy_(out.detach())
    small_p.requires_grad = True

train(large_model, tokenizer, train_tokens=9e9, eval_batch=1000, prompt=prompt)
torch.save(small_model.state_dict(), 'gpt2_small_40m.ckpt')
s = sample(small_model, tokenizer, prompt=prompt, temperature=0.8, top_k=500, max_tokens=200)
print(s)