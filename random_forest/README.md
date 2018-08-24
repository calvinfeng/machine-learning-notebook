# Random Forest
For full explanation, please look at the `random_forest.ipynb` notebook.

## Pruning
Pruning is the inverse of splitting. Grow the tree fully until leaf nodes have minimum impurity.
Then all pairs of leaf nodes are considered for elimination. Any pair whose elimination yields a 
satisfactory (small) increase in impurity is eliminated, and the common antecedent node is declared
as leaf node.