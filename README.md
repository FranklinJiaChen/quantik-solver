# quantik-solver

Solves the combinatorial game [Quantik](https://boardgamegeek.com/boardgame/286295/quantik)

## Methodology

1. Represent the board using a 2d array.
2. Normalize the board using
   [sudoku symmetry group](https://pi.math.cornell.edu/~mec/Summer2009/Mahmood/Symmetry.html)
   to reduce the solution space. (# states after move 2 reduced from 3840 --> 7)

   Note that reflections are not included
   as separate elements in the symmetry group,
   as they are identically created through
   permutations of rows, columns, bands, and stacks.
   This approach differs from the referenced link,
   which includes reflections as part of the symmetry group.

3. Solve using minimax and memoization.

## Results
Player 2 is able to force a win through perfect play.
