### Multiple Sequence Alignment
- The algorithm aligns all pairs of sequences using the Needleman-Wunsch global sequence alignment algorithm. Then it aligns the profile of the pair with the highest pairwise alignment score with the third sequence. 
- The scoring function:
    - Match reward of 1 
    - Indel/mismatch penalty of 0
