# Kris Kringle

Given a set of grouped individuals, I need to pair everyone up for Kris Kringle such that no two
individuals are matched within the same group.

To formalise this, let `n_i^j` (`i \in {1, ..., N}`, `j \in {1, ..., M}`) denote the `i`'th
individual, where `j` denotes the group that the individual belongs to.
Find a graph `G = {(n_i^j, n_k^l)}` such that:
* `j != l` for all `g \in G`
* `\all n_i^j`, `\exists g_1, g_2 \in G` where `n_i^j \in g1, g2` and `g1 != g2`

If we consider the adjacency matrix `A` for this graph `G`, we note the following properties:
* The row sums of `A` are all equal to 1
* The column sums of `A` are all equal to 1
* `A` has a block diagonal structure `A ~ diag(\Lambda_1, \Lambda_2, ...)` where `\Lambda_j`
  denotes a square zero matrix with size equal to the number of individuals in group `j` (denoted `M_j`)

Question: Under what conditions does a solution `G` exist?

Proposition: Let `\gamma` denote the number of non-block elements of `A` (given by `N^2 - \sum_j
M_j`). Then a solution `A` does not exist if and only if `(N + 1) % \gamma = 0`

I can't seem to prove this.. Is there a counterexample?
