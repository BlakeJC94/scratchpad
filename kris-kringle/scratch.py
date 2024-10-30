
# Given groups (a1, b1, ...), (a2, b2, ...), ...
# Create a pairing such that none of the pairs occur in the same group

import random
from copy import deepcopy
from itertools import product

# %%
elements = [
    ('blake', 0),
    ('claire', 2),
    ('grace', 2),
    ('miriam', 3),
    ('nathan', 1),
    ('rachel', 1),
    ('zoe', 0),
]

# %%
all_possible_pairs = [(a, b) for a, b in product(elements, elements) if a[0] != b[0] and a[1] != b[1]]


# %%

# pairs = []

# idx = random.randint(0, len(all_possible_pairs) - 1)
# selected_pair = all_possible_pairs[idx]
# pairs.append(selected_pair)

# # %%
# filtered_pairs = [(a,b) for a,b in all_possible_pairs if a[0] != selected_pair[0][0] and set((a, b)) not in [set(p) for p in pairs]]

# %%
for i in range(1000):
    try:
        passed = False
        pairs = []
        filtered_pairs = deepcopy(all_possible_pairs)
        for i in range(0, len(elements)):
            if i > 0:
                foo = [(a,b) for a,b in filtered_pairs if a[0] == selected_pair[1][0]]
            else:
                foo = filtered_pairs

            idx = random.randint(0, len(foo) - 1)
            selected_pair = foo[idx]
            pairs.append(selected_pair)

            filtered_pairs = [(a,b) for a,b in filtered_pairs if a[0] != selected_pair[0][0] and set((a, b)) not in [set(p) for p in pairs]]

        passed = True
        break
    except Exception:
        continue

if passed:
    print("WOO")



# %%

pairs = []

idx = random.randint(0, len(filtered_pairs) - 1)

selected_pair = filtered_pairs[idx]
filtered_pairs = deepcopy(valid_pairs)
for i in range(len(elements)):
    idx = random.randint(0, len(filtered_pairs) - 1)
    selected_pair = filtered_pairs[idx]


    filtered_pairs = [{a, b} for a, b in filtered_pairs]






# %%
# %%
# %%
# %%
def main():
    pairs = []
    all_possible_pairs = [{a, b} for a, b in product(elements)]

    # Filter invalid pairs from all possible pairs
    valid_pairs = []
    for pair in all_possible_pairs:
        element1, element2 = pair
        name1, ind1 = element1
        name2, ind2 = element2
        if name1 == name2 or ind1 == ind2:
            continue
        valid_pairs.append(pair)

    ...


    for (a, b) in pairs:
        print(f"{a} has {b} for kris kringle")


# remaining_elements = deepcopy(elements)
# for element in elements:
#     if any(element == e for e in pair for pair in pairs)


