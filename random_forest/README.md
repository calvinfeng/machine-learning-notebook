# Random Forest
For full explanation, please look at the Jupyter notebook.

## Pseudo-code
1. Randomly select `k` features from total `K` features, where `k << K`
2. Construct a decision tree using `k` features and give it a threshold for splitting
    * Minimum sample split, i.e. minimum number of samples required to split an internal node
    * Minimum samples leaf, i.e. minimum number of samples required to be a leaf node
    * Max leaf nodes
    * Minimum purity increase
    * Maximum depth
3. Build a forest by repeating steps 1 to 3 for `n` times to create `n` number of tree.
4. Make prediction through majority vote.

## Pruning
Pruning is the inverse of splitting. Grow the tree fully until leaf nodes have minimum impurity.
Then all pairs of leaf nodes are considered for elimination. Any pair whose elimination yields a 
satisfactory (small) increase in impurity is eliminated, and the common antecedent node is declared
as leaf node.