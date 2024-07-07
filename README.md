# quantik-solver

Solves the combinatorial game "Qauntik"

## Methodology

1. Represent the board using a 2d array.
2. Normalize the board rotations/reflections (dihedral group D_4)
   and each player'shapes (symmetric group S_4)
   to reduce the solution space
3. Solve using minimax and memoization