# %% Load the data

data_location = "./data/tinyshakespeare/input.txt"

with open(data_location, "r", encoding="utf-8") as f:
    text = f.read()

len(text)
print(text[:1000])

# %% Get vocab (poor persons tokenizer)
chars = sorted(set(text))
vocab_size = len(chars)

print("".join(chars))
print(vocab_size)


# %% Create encoder/decoder
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

print(encode("hii there"))
print(decode(encode("hi there")))


# %% Encode data into torch tensor
import torch

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])


# %% Create 90/10 train/val split of data
n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

# %% Define block size (context window)
block_size = 8
print(train_data[: block_size + 1])

# %% Illustrate the training inputs to the transformer
x = train_data[:block_size]
y = train_data[1 : block_size + 1]
for t in range(block_size):
    context = x[: t + 1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

# %% Get batches
torch.manual_seed(1337)
batch_size = 4
block_size = 8


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


xb, yb = get_batch("train")
print("inputs:")
print(xb.shape)
print(xb)
print("targets:")
print(yb.shape)
print(yb)

print("----")

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, : t + 1]
        target = yb[b, t]
        print(f"when input is {context.tolist()}, the target: {target}")

# %%
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


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
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

idx = torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))


# %%

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size = 32

for steps in range(1000):
    xb, yb = get_batch("train")

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

# %%
idx = torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=300)[0].tolist()))

# %% The trick in self-attention
torch.manual_seed(1337)
B, T, C = 4,8,32
x = torch.randn(B, T, C)
print(x.shape)

# %%
# we want x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B,T,C))  # bag of words
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]
        xbow[b,t] = torch.mean(xprev, 0)

# %%
torch.manual_seed(42)
a = torch.tril(torch.ones(3,3))
a = a / torch.sum(a, dim=1, keepdim=True)
b = torch.randint(0, 10, (3,2)).float()
c = a @ b
print(f"a=\n{a}")
print("---")
print(f"b=\n{b}")
print("---")
print(f"c=\n{c}")

# %%

wei = torch.tril(torch.ones(T,T))
wei = wei / torch.sum(wei, dim=1, keepdim=True)
xbow2 = wei @ x
print(f"{torch.allclose(xbow, xbow2)=}")

# %%
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x

# %% Self attention
torch.manual_seed(1337)
B, T, C = 4,8,32
x = torch.randn(B, T, C)

# single head of self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)  # ~ (B, T , head_size)
q = query(x)  # ~ (B, T , head_size)

wei = q @ k.transpose(-2, -1)  # (B, T , head_size) @ (B, head_size, T) ~ (B,T,T)

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = wei / torch.sqrt(head_size)
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v

# %%
# %%
