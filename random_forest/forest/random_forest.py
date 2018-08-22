import numpy as np
from decision_tree import DecisionTree
import pdb


class RandomForest(object):
    def __init__(self, n_samples=1000, n_features=4):
        """Construct a random forest using decision tree

        There are couple missing features for this implementation.
        - Minimum sample split
        - Minimum sample leaf
        - Minimum purity increase
        - Maximum leaf nodes
        - Maximum depth

        Args:
            n_samples (int): Number of decision tree in the forest
            n_features (int): Number of features that is used in each tree
        """
        self.n_samples = 1000
        self.n_features = 4
        self.forest = []

    def fit(self, col_names, rows):
        """
        Currently tree bagging (bootstrap aggregating) is not implemented, every tree receives the
        same training set but they are trained on different features.
        """
        # Last el is alwasy label for this setup.
        for _ in range(self.n_samples):
            k_indices = np.random.choice(len(col_names) - 1, self.n_features, replace=False)
            tree = DecisionTree(feat_indices=k_indices)
            tree.fit(col_names, rows)
            self.forest.append(tree)

    def predict(self, row):
        """Gives prediction by majority vote from the forest.

        Args:
            row (list): A row of data
        """
        label_vote = dict()
        for i in range(len(self.forest)):
            result = self.forest[i].predict(row)
            label = max(result, key=result.get)
            
            if label_vote.get(label, None) is None:
                label_vote[label] = 0

            label_vote[label] += 1
        
        return max(label_vote, key=result.get)
