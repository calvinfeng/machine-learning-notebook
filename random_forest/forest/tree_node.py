from split_rule import SplitRule
from util import partition, gini, impurity_reduction, entropy, info_gain


class TreeNode(object):
    """Tree node can either be a decision node or a prediction node.

    Every leaf node is a prediction node, which carries the logic of giving a prediction with 
    quantifiable confidence. Every other node is a decision node where it has two children. We are
    assuming a binary tree implementation for decision tree.
    """

    def __init__(self, X, attr_names):
        self.X = X
        self.attr_names = attr_names


    def seek_split_rule(self, criterion='gini'):
        if criterion == 'gini':
            metric = gini
            eval_score = impurity_reduction
        elif criterion == 'entropy':
            metric = entropy
            eval_score = info_gain
        else:
            raise ValueError('%s is not a valid partition criterion' % criterion)

        best_rule = None
        best_score = 0
        curr_metric_val = metric(self.X)

        for i in range(len(self.attr_names) - 1):
            # Extract unique values from dataset in a given feature/column.
            values = set([x[i] for x in self.X])

            for val in values:
                rule = SplitRule(self.attr_names[i], i, val)

                # Partition the current dataset and check if everything landed on one side. If so, 
                # this is a bad partition, don't consider it.                
                true_set, false_set = partition(self.X, rule)
                if len(true_set) == 0 or len(false_set) == 0:
                    continue
                
                score = eval_score([true_set, false_set], curr_metric_val, len(self.X))
                if score >= best_score:
                    best_score, best_rule = score, rule

        return best_score, best_rule


    def split(self):
        pass


if __name__ == '__main__':
    header = ['color', 'diameter', 'label']

    training_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 3, 'Apple'],
        ['Red', 1, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon']
    ]

    root = TreeNode(training_data, header)
    print root.seek_split_rule(criterion='entropy')
