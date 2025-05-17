# bitsolvers
Various frameworks for solving twisty puzzles that can store puzzle states compactly in integers, using bitwise operators to perform moves. Built in python and uses numpy when suitable/possible.

Current solvers added:
### CPFB
This one stores the permutations of one center, all corners and three edges as well as the orientation of 3 edges in 61 bits, so np.uint64 can be used.
