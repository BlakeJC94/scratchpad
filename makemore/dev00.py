data = "./data/names.txt"

with open(data, "r") as f:
    words = f.read().splitlines()

words[:10]

# %%
len(words)

# %%
print(min(len(w) for w in words))
print(max(len(w) for w in words))

# %%
b = {}
for w in words:
    chs = ['<S>', *list(w), '<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

# %%
sorted(b.items(), key=lambda v: -v[1])

# %%
chars = sorted(set("".join(words)))
stoi = {c: i+1 for i, c in enumerate(chars)}
stoi['.'] = 0

# %%
import torch

N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.', *list(w), '.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# %%
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Qt5Agg')
# plt.imshow(N)
# plt.show()

# %%
p = N[0, :].float()
p = p / p.sum()
p


# %%
itos = {i: s for s, i in stoi.items()}

# %%
torch.manual_seed(42)
ix = torch.multinomial(p, num_samples=1, replacement=True).item()
itos[ix]

# %%
P = N.float()
P /= P.sum(1, keepdim=True)

# %%
g = torch.Generator().manual_seed(1234)

for i in range(5):
    ix = 0
    out = []
    while True:
        p = P[ix].float()
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print("".join(out))


# %% Loss function
log_likelihood = 0.
n = 0
for w in words:
    chs = ['.', *list(w), '.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        # print(f"{ch1}{ch2}: {prob:.4f} {logprob:.4f}")

print(f"{log_likelihood=}")
nll = -log_likelihood
print(f"{nll=}")
print(f"{nll/n=}")

# %%
# create training set of bigrams
xs, ys = [], []

for w in words[:1]:
    chs = ['.', *list(w), '.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
print(f"{xs=}")
print(f"{ys=}")

# %%
import torch.nn.functional as F

xenc = F.one_hot(xs, num_classes=27).float()

# %%

W = torch.randn(27, 27)
xenc @ W

# %%
logits = (xenc @ W)  # log counts
counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)

# %%
nlls = torch.zeros(5)
for i in range(5):
    x = xs[i].item()
    y = ys[i].item()
    print("--------")
    print(f"bigram example {i+1}: {itos[x]}{itos[y]}")
    print(f"input: {x}")

# %%
# %%
nlls = torch.zeros(5)
for i in range(5):
    x = xs[i].item()
    y = ys[i].item()
    print("--------")
    print(f"bigram example {i+1}: {itos[x]}{itos[y]}")
    print(f"input: {x}")
    print(f"output probs from neural net:")
    print(f"{probs[i]}")
    print(f"actual: {y}")
    p = probs[i, y]
    print(f"prob assigned to correct char: {p.item()}")
    logp = torch.log(p)
    nll = -logp
    nlls[i] = nll

print(f"========")
print(f"avg neg log likelihood {nlls.mean().item()}")


# %%
ys

# %% create training set of bigrams
xs, ys = [], []

for w in words:
    chs = ['.', *list(w), '.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs).to("cuda")
ys = torch.tensor(ys).to("cuda")
num = xs.nelement()
print(f"number of examples: {num}")

# %% init network
g = torch.Generator(device="cuda").manual_seed(1234)
W = torch.randn((27, 27), generator=g, requires_grad=True, device=torch.device("cuda:0"))

# %% gradient descent
for k in range(100):

    # forward pass
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = (xenc @ W)  # log counts
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W ** 2).mean()
    print(loss.item())

    # backward pass
    W.grad = None  # set to 0
    loss.backward()

    # Update
    W.data += -50 * W.grad

# %% evaluate
for i in range(5):
    ix = 0
    out = []
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float().cuda()
        logits = (xenc @ W)  # log counts
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True)

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print("".join(out))
