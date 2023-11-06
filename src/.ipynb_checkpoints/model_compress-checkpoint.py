import torch
from tokenizer import Tokenizer
from model import Transformer
from train import get_modelargs, get_prompt, sample, train

prompt = "JULIUS\n"

large_model_redf = 32
tokenizer = Tokenizer('/media/apurva/nvme/mistral-7B-v0.1/tokenizer.model')
large_model_args = get_modelargs(reduction_factor=large_model_redf, vocab_size=tokenizer.vocab_size)
large_model = Transformer(args=large_model_args).cuda()

print(large_model)

# train(large_model, tokenizer, batches=5000, eval_batch=1000, prompt=prompt)
# torch.save(large_model.state_dict(), 'full_model_rf4.ckpt')

saved_rf4_model = torch.load('full_model_rf4.ckpt')
large_model.load_state_dict(saved_rf4_model)
s = sample(large_model, tokenizer, prompt=prompt, temperature=0.8, top_k=500, max_tokens=200)
print(s)

small_model_args = get_modelargs(reduction_factor=large_model_redf, vocab_size=tokenizer.vocab_size)
small_model_args.dim = small_model_args.dim // 2
small_model_args.hidden_dim = small_model_args.hidden_dim // 2
small_model_args.head_dim = small_model_args.head_dim // 2

small_model = Transformer(args=small_model_args).cuda()
s = sample(small_model, tokenizer, prompt=prompt, temperature=0.8, top_k=500, max_tokens=200)
print(s)

print("largeparams", sum(p.numel() for p in large_model.parameters()))

print("smallparams", sum(p.numel() for p in small_model.parameters()))

for (large_name, large_p), (small_name, small_p) in zip(large_model.named_parameters(), small_model.named_parameters()):

    print(small_name, large_p.shape, small_p.shape)

    small_p.requires_grad = False
    large_p.requires_grad = False

    assert large_name == small_name
    small_p *= 0
    if len(large_p.shape) == 2:
        small_p += large_p[0:small_p.shape[0], 0:small_p.shape[1]]

    elif len(large_p.shape) == 1:
        small_p += large_p[0:small_p.shape[0]]
        small_p += large_p[small_p.shape[0]:]

    else:
        print('missed params', large_name)
    small_p.requires_grad = True

train(small_model, tokenizer, batches=1000, eval_batch=100)
s = sample(small_model, tokenizer, prompt=prompt, temperature=0.8, top_k=500, max_tokens=200)
print(s)