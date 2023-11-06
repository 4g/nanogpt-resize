import glob
import random

import torch
import numpy as np

torch.set_default_device('cuda')
# torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

from tqdm import tqdm
from model import Transformer, ModelArgs
from nanogpt_model import GPT, GPTConfig
from tokenizer import Tokenizer, CharTokenizer


def get_modelargs(reduction_factor=2, vocab_size=32000):
    return ModelArgs(dim=4096//reduction_factor,
              hidden_dim=14336//reduction_factor,
              head_dim=128//reduction_factor,
              n_heads=32//reduction_factor,
              vocab_size=vocab_size,
              n_layers=32//reduction_factor,
              norm_eps=1e-5,
              max_seq_len=4096,
              dtype=torch.float32)


MODELARGS = {
    '7B' : get_modelargs(reduction_factor=1),
    '1B' : get_modelargs(reduction_factor=2),
    '150m' : get_modelargs(reduction_factor=4),
    '40m' : get_modelargs(reduction_factor=8),
    '10m' : get_modelargs(reduction_factor=16),
    'tiny': get_modelargs(reduction_factor=32)
}

def make_dataset(batch_size, seq_len, tokenizer):
    import pandas

    files = glob.glob("../data/*.parquet")
    data = []
    for file in files:
        df = pandas.read_parquet(file)
        df = df['content'].tolist()
        data.extend(df)

    print(len(data))

    token_batch = []
    num_batch_tokens = batch_size * seq_len
    for elem in data:
        tokens = tokenizer.encode(elem, add_bos=True, add_eos=True)
        token_batch.extend(tokens)
        if len(token_batch) > num_batch_tokens:
            arr = np.asarray(token_batch[:num_batch_tokens])
            arr = arr.reshape((batch_size, seq_len))
            yield arr
            token_batch = token_batch[num_batch_tokens:]


def make_dataset_shakes(batch_size, seq_len, tokenizer):
    text = open("../data/shakespeare.txt").read()

    token_batch = []
    num_batch_tokens = batch_size * seq_len
    tokens = tokenizer.encode(text)
    while True:
        rand_index = lambda: random.randint(0, len(tokens) - seq_len)
        for i in range(batch_size):
            start = rand_index()
            end = start + seq_len
            token_batch.append(tokens[start: end])

        yield np.asarray(token_batch)
        token_batch = []

def make_dataset_openwebtext(batch_size, seq_len, shard='val'):
    tokens = np.memmap(f'/media/apurva/nvme/openwebtext/{shard}.bin',  dtype=np.uint16, mode='r')

    token_batch = []
    while True:
        rand_index = lambda: random.randint(0, len(tokens) - seq_len + 1)
        for i in range(batch_size):
            start = rand_index()
            end = start + seq_len + 1
            token_batch.append(tokens[start: end])

        token_batch = np.asarray(token_batch, dtype=np.int32)

        input_token_ids = token_batch[:, :seq_len]
        output_token_ids = token_batch[:, 1:]
        input_token_ids = torch.LongTensor(input_token_ids).cuda()
        output_token_ids = torch.LongTensor(output_token_ids).cuda()
        yield input_token_ids, output_token_ids

        token_batch = []


def sample(model, tokenizer, prompt, temperature=0.8, top_k=500, max_tokens=200):
    with torch.inference_mode() and torch.no_grad():
        tokens = tokenizer.encode(prompt)
        print("Prompt", tokens)
        for i in range(max_tokens):
            input_tokens = torch.LongTensor([tokens]).cuda()
            logits, loss = model(input_tokens)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            best_token = idx_next.detach().cpu().numpy()
            best_token = int(best_token[0][0])
            tokens = tokens + [best_token]
            # if best_token == tokenizer.bos_id or best_token == tokenizer.eos_id:
            #     break
    # print(tokens)
    return tokenizer.decode(tokens)


def train(model, tokenizer, train_tokens=5000, eval_batch=1000, prompt="\n"):
    print(prompt)

    optimizer = model.configure_optimizers(weight_decay=0.1,
                                           betas=(0.9, 0.95),
                                           learning_rate=3e-4,
                                           device_type='cuda')

    num_params = sum(param.numel() for name, param in model.named_parameters())
    print("Num params", num_params)

    bsz = 64
    sql = 256
    n_batches = int(train_tokens / (bsz * sql))
    print("Number of batches in training =", n_batches)

    model.compile()
    token_generator = make_dataset_openwebtext(batch_size=bsz, seq_len=sql+1, shard='train')
    val_token_generator = make_dataset_openwebtext(batch_size=bsz, seq_len=sql+1, shard='val')

    eval_iters = 100
    val_batches = [next(val_token_generator) for i in range(eval_iters)]

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = torch.zeros(eval_iters)
        for k, (X, Y) in enumerate(val_batches):
            with torch.autocast(dtype=torch.bfloat16, device_type='cuda'):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out = losses.mean()
        model.train()
        return out

    for i in tqdm(range(n_batches), disable=False):
        input_token_ids, output_token_ids = next(token_generator)
        with torch.autocast(dtype=torch.bfloat16, device_type='cuda'):
            logits , loss = model(input_token_ids, output_token_ids)

        loss.backward()

        if i % eval_batch == 0:
            eval_loss = estimate_loss()
            print("train loss:", loss, "eval loss:", eval_loss)
            s = sample(model, tokenizer, prompt)
            print(s)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        optimizer.zero_grad(set_to_none=True)

    return model