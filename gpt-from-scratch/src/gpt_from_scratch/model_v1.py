import torch
from torch import nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

# %% Define block size (context window)
block_size = 8

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
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

# %% Create model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx ~ (B, T)
        for _ in range(max_new_tokens):
            logits, _loss = self(idx)  # Get preds
            logits = logits[:, -1, :]  # ~ (B, C), focus on last timestep
            probs = F.softmax(logits, dim=-1)  # ~ (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # ~ (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # ~ (B, T+1)
        return idx


m = BigramLanguageModel(vocab_size)
m = m.to(device)

# %% Optimise
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# %% Generate from model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

