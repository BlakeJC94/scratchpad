import random
from copy import deepcopy
from itertools import product

elements = [
    ('blake', 0),
    ('claire', 2),
    ('grace', 2),
    ('miriam', 3),
    ('nathan', 1),
    ('rachel', 1),
    ('zoe', 0),
]

def main():
    all_possible_pairs = [(a, b) for a, b in product(elements, elements) if a[0] != b[0] and a[1] != b[1]]
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
        print("Kris kingle pairings:")
        for ((a,_), (b,_)) in pairs:
            print(f"  {a: <6} has {b: <6}")
    else:
        raise ValueError("Nope")

if __name__ == "__main__":
    main()

