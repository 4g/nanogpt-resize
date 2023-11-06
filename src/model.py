from torch import nn
from dataclasses import dataclass
import inspect
import torch
torch.manual_seed(0)


@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    norm_eps: float
    vocab_size: int
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0
    embd_drop: float = 0.0
    max_seq_len:int = 4096
    bias = False
    max_batch_size: int = 0
    dtype: torch.dtype = None


class CausalSelfAttention(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.dim, 3 * config.dim, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.mlp_dropout)
        self.n_head = config.n_heads
        self.n_embd = config.dim
        self.dropout = config.mlp_dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = True


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype=torch.float32):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.dim = args.dim

        self.norm1 = RMSNorm(dim=args.dim, eps=args.norm_eps, dtype=args.dtype)
        self.attention = CausalSelfAttention(config=args)

        self.norm2 = RMSNorm(dim=args.dim, eps=args.norm_eps, dtype=args.dtype)

        self.gate_proj = nn.Linear(args.dim, args.hidden_dim, bias=False, dtype=args.dtype)
        self.up_proj = nn.Linear(args.dim, args.hidden_dim, bias=False, dtype=args.dtype)
        self.down_proj = nn.Linear(args.hidden_dim, args.dim, bias=False, dtype=args.dtype)

        self.residual_drop = nn.Dropout(args.mlp_dropout)

        self.silu = nn.functional.silu
        self.dtype = args.dtype

    def forward(self, x):

        x = self.norm1(x)
        attn_output = self.attention(x)

        hidden = x + attn_output
        hidden_norm = self.norm2(hidden)
        output = self.down_proj(self.silu(self.gate_proj(hidden_norm)) * self.up_proj(hidden_norm))
        output = hidden + output

        output = self.residual_drop(output)
        return output


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=args.vocab_size,
                                       embedding_dim=args.dim,
                                       dtype=args.dtype)

        self.pos_embeddings = nn.Embedding(num_embeddings=args.max_seq_len,
                                           embedding_dim=args.dim,
                                           dtype=args.dtype)

        self.embd_dropout = nn.Dropout(args.embd_drop)

        # make list of blocks a torch object so it can be identified as a part of model
        self.layers = torch.nn.ModuleList([TransformerBlock(args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps, dtype=args.dtype)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False, dtype=args.dtype)

        # tie weights of lm_head with embeddings
        self.embeddings.weight = self.lm_head.weight

    def forward(self, token_ids, targets=None):
        pos = torch.arange(0, token_ids.size()[1], dtype=torch.long, device=token_ids.device).unsqueeze(0)
        embeds = self.embeddings(token_ids) + self.pos_embeddings(pos)
        embeds = self.embd_dropout(embeds)
        for idx, layer in enumerate(self.layers):
            embeds = layer(embeds)
        embeds = self.norm(embeds)

        loss = None
        if targets is not None:
            logits = self.lm_head(embeds)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(embeds[:, [-1], :])
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
