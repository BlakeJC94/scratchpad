import torch
import torch.nn.functional as F
import matplotib.pyplot as plt

# %% Read in all the words
with open("./data/names.txt") as f:
    words = f.read().splitlines()

words[:10]
len(words)

# %%
chars = sorted(set("".join(words)))
stoi = {s: i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
print(itos)

# %%
block_size = 3  # context length

X, Y = [], []
for w in words[:5]:
    print(w)
    context = [0] * block_size
    for ch in (w + "."):
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        print("".join(itos[i] for i in context), "--->", itos[ix])
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)


# %%
C = torch.randn((27, 2))
# %%
F.one_hot(torch.tensor(5), num_classes=27).float() @ C
# %%
C[X]
