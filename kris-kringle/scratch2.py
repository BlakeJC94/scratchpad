import numpy as np
from scipy import linalg as la


# %%
A = np.array(
    [
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 1., 0.],
        [0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1.],
        [1., 0., 0., 0., 0.],
    ]
)

B = la.block_diag(
    np.ones((2, 2)),
    np.ones((2, 2)),
    np.ones((1, 1)),
)
Bc = np.ones((5,5)) - B

one_vec = np.ones((5, 1))

# %%
A @ Bc

# %%
A @ B

# %%
B @ A

# %%

A @ one_vec
one_vec.T @ A
one_vec.T @ A @ one_vec  # [5]

# %%
# %%
