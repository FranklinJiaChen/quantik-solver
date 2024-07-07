# quantik-solver

Solves the combinatorial game "Qauntik"

## Methodology

1. Represent the board using a 2d array.
2. Normalize the board using
   [sudoku symmetry group](https://pi.math.cornell.edu/~mec/Summer2009/Mahmood/Symmetry.html)
   to reduce the solution space.

   Note that reflections are not included
   as separate elements in the symmetry group,
   as they are identical created through
   permutations of rows, columns, bands, and stacks.
   This approach differs from the referenced link,
   which includes reflections as part of the symmetry group.

3. Solve using minimax and memoization