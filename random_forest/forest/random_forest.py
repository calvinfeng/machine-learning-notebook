import numpy as np
from decision_tree import DecisionTree


class RandomForest(object):
    def __init__(self, n_samples=1000, n_features=4):
        self.n_samples = 1000
        self.n_features = 4
        self.forest = []
        # TODO:
        # Min. sample split
        # Min. sample leaf
        # Min. purity increase
        # Max. leaf nodes
        # Max. depth

    def fit(self, col_names, rows):
        """
        Currently tree bagging is not implemented, every tree receives the same training set but
        they are trained on different features.
        """
        for _ in range(self.n_samples):
            # Last el is alwasy label for this setup.
            k_indices = np.random.choice(len(rows) - 1, self.n_features)
            tree = DecisionTree(feat_indices=k_indices)
            tree.fit(col_names, rows)
            self.forest.append(tree)

    def predict(self, row):
        """
        Performs majority vote from each tree.
        """
        label_vote = dict()
        for i in range(len(self.forest)):
            result = self.forest[i].predict(row)
            label = max(result, key=result.get)
            
            if label_vote.get(label, None) is None:
                label_vote[label] = 0

            label_vote[label] += 1
        
        return label_vote
