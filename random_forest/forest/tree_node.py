from split_rule import SplitRule
from util import partition, gini, purity_gain, entropy, info_gain


class TreeNode(object):
    """Tree node can either be a decision node or a prediction node.

    Every leaf node is a prediction node, which carries the logic of giving a prediction with 
    quantifiable confidence. Every other node is a decision node where it has two children. We are
    assuming a binary tree implementation for decision tree.
    """

    def __init__(self, column_names, X, feat_indices=None):
        """Construct a decision tree node

        Args:
            X (list): A multi-dimensional list, of shape (N, D) where D is the number of columns.
            column_names ([]string): A list of column names.
            feat_indices ([]int): A list of indices that is used for splitting.
        """
        self.X = X
        self.rule = None
        self.column_names = column_names
        self.true_branch = None
        self.false_branch = None
        
        self.feat_indices = range(len(column_names) - 1)
        if feat_indices is not None:
            self.feat_indices = feat_indices

    @property
    def prediction(self):
        if self.true_branch is not None and self.false_branch is not None:
            return None 

        N = len(self.X)
        prob = dict()
        for row in self.X:
            label = row[-1]
            if label not in prob:
                prob[label] = 0.0
            
            prob[label] += 1

        for label in prob:
            prob[label] /= N
        
        return prob

    def seek_split_rule(self, criterion='gini'):
        if criterion == 'gini':
            metric = gini
            eval_gain = purity_gain
        elif criterion == 'entropy':
            metric = entropy
            eval_gain = info_gain
        else:
            raise ValueError('%s is not a valid partition criterion' % criterion)

        best_gain = 0
        current_metric_val = metric(self.X)

        for i in self.feat_indices:
            # Extract unique values from dataset in a given feature/column.
            values = set([x[i] for x in self.X])

            for val in values:
                rule = SplitRule(self.column_names[i], i, val)

                # Partition the current dataset and check if everything landed on one side. If so, 
                # this is a bad partition, don't consider it.                
                true_set, false_set = partition(self.X, rule)
                if len(true_set) == 0 or len(false_set) == 0:
                    continue
                
                gain = eval_gain([true_set, false_set], current_metric_val, len(self.X))
                if gain >= best_gain:
                    best_gain, self.rule = gain, rule

    def split(self):
        self.seek_split_rule()

        if self.rule is None:
            return None

        true_set, false_set = partition(self.X, self.rule)
        self.true_branch = TreeNode(self.column_names, X=true_set)
        self.false_branch = TreeNode(self.column_names, X=false_set)

        return self.true_branch, self.false_branch                


    