import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader

from torch.optim import Adam
from torchkge.models import TransEModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss # , DataLoader  # TODO: Why dataloader?
from torchkge.utils.datasets import load_fb15k


# writer = SummaryWriter(f"runs/toynet")
# writer.add_scalar("Loss/train", epoch_loss, global_step)

# %% Load dataset
kg_train, kg_val, kg_test = load_fb15k()

# %% Define hparams
emb_dim = 100
lr = 4e-3
n_epochs = 1000
b_size = 32768
margin = 0.5
device = torch.device("mps")

# %% Define model and criterion
model = TransEModel(
    emb_dim,
    kg_train.n_ent,
    kg_train.n_rel,
    dissimilarity_type="L2",
)
criterion = MarginLoss(margin)  # TODO what is margin?

# %% Move to device
model = model.to(device)
criterion = criterion.to(device)

# %% Define optimizer
optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

# %% Define dataloader
sampler = BernoulliNegativeSampler(kg_train)
dataloader = DataLoader(kg_train, batch_size=b_size)

# %% Train model
iterator = tqdm(
    range(n_epochs),
    unit="epoch",
)
writer = SummaryWriter(f"runs/transemodel")

global_step = 0
for epoch in iterator:
    running_loss = 0.0

    for i, batch in enumerate(dataloader):
        h, t, r = batch
        n_h, n_t = sampler.corrupt_batch(h, t, r)
        h = h.to(device)
        t = t.to(device)
        r = r.to(device)
        n_h = n_h.to(device)
        n_t = n_t.to(device)

        optimizer.zero_grad()

        pos, neg = model(h, t, r, n_h, n_t)
        loss = criterion(pos, neg)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        global_step += 1

    epoch_loss = running_loss/len(dataloader)
    writer.add_scalar("Loss/train", epoch_loss, global_step)
    iterator.set_description(
        "Epoch {} | mean loss: {:.5f}".format(
            epoch + 1,
            epoch_loss
        )
    )

