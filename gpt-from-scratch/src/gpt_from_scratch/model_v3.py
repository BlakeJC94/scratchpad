from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

batch_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
# ----

torch.manual_seed(1337)

# %% Load the data
data_location = "./data/tinyshakespeare/input.txt"
with open(data_location, "r", encoding="utf-8") as f:
    text = f.read()

# %% Get vocab (poor persons tokenizer)
chars = sorted(set(text))
vocab_size = len(chars)

# %% Create encoder/decoder
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

# %% Encode data into torch tensor and create 90/10 train/val split of data
data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# %% Data loading
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# %% Evaluation
@torch.no_grad()
def estimate_loss(model, eval_iters):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# %% Create model
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)  # ~ (B, T , head_size)
        q = self.query(x)  # ~ (B, T , head_size)

        # Compute affinities
        wei = q @ k.transpose(-2, -1) * (self.head_size**-0.5)  # ~ (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)]
        )
        self.key = nn.Linear(n_embd, head_size * num_heads, bias=False)
        self.query = nn.Linear(n_embd, head_size * num_heads, bias=False)
        self.value = nn.Linear(n_embd, head_size * num_heads, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape

        xhat = []

        k = self.key(x).split(self.num_heads, dim=-1)  # ~ [(B, T , head_size) * num_heads]
        q = self.query(x).split(self.num_heads, dim=-1)  # ~ [(B, T , head_size) * num_heads]

        for _ in range(len(self.heads)):

            k = self.key(x)  # ~ (B, T , head_size)
            q = self.query(x)  # ~ (B, T , head_size)

            # Compute affinities
            wei = q @ k.transpose(-2, -1) * (self.head_size**-0.5)  # ~ (B,T,T)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)

            v = self.value(x)
            out = wei @ v
            xhat.append(out)

        x = self.proj(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


@dataclass
class ModelConfig:
    block_size: int = 256
    vocab_size: int = 256
    n_embd: int = 384
    n_layer: int = 6
    n_head: int = 6
    dropout: float = 0.2


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(
                    config.n_embd,
                    n_head=config.n_head,
                    block_size=config.block_size,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        # idx ~ (B, T), targets ~ (B, T)
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # ~(B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # ~(T,C)
        x = tok_emb + pos_emb  # ~ (B, T, C)

        x = self.blocks(x)  # ~ (B, T, C)
        x = self.ln_f(x)  # ~ (B,T,C)
        logits = self.lm_head(x)  # ~(B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx ~ (B, T)
        for _ in range(max_new_tokens):
            idx_cond = idx[
                :, -self.config.block_size :
            ]  # Crop to last block_size tokens
            logits, _loss = self(idx_cond)  # Get preds
            logits = logits[:, -1, :]  # ~ (B, C), focus on last timestep
            probs = F.softmax(logits, dim=-1)  # ~ (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # ~ (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # ~ (B, T+1)
        return idx


config = ModelConfig(
    block_size=256,
    vocab_size=vocab_size,
    n_embd=384,
    n_layer=6,
    n_head=6,
    dropout=0.2,
)
m = LanguageModel(config)
m = m.to(device)

# %% Optimise
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(m, eval_iters)
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# %% Generate from model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
