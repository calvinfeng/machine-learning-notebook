from split_rule import SplitRule
from util import partition, gini, purity_gain, entropy, info_gain


class TreeNode(object):
    """Tree node can either be a decision node or a prediction node.

    Every leaf node is a prediction node, which carries the logic of giving a prediction with 
    quantifiable confidence. Every other node is a decision node where it has two children. We are
    assuming a binary tree implementation for decision tree.
    """

    def __init__(self, attr_names, X):
        self.X = X
        self.attr_names = attr_names
        self.true_branch = None
        self.false_branch = None

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

        self.rule = None
        best_gain = 0
        current_metric_val = metric(self.X)

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
                
                gain = eval_gain([true_set, false_set], current_metric_val, len(self.X))
                if gain >= best_gain:
                    best_gain, self.rule = gain, rule

    def split(self):
        self.seek_split_rule()

        if self.rule is None:
            return None

        true_set, false_set = partition(self.X, self.rule)
        self.true_branch = TreeNode(self.attr_names, X=true_set)
        self.false_branch = TreeNode(self.attr_names, X=false_set)

        return self.true_branch, self.false_branch                


def classify(node, x):
    """
    Args:
        root (TreeNode):
        x (numpy.ndarray): One-dimensional array, which is a row of data
    """
    if node.rule is None:
        return node.prediction
    
    if node.rule.match(x):
        return classify(node.true_branch, x)
    
    return classify(node.false_branch, x)


def build_tree(root):
    if root.split() is not None:
        build_tree(root.true_branch)
        build_tree(root.false_branch)


def print_tree(root, spacing=''):
    if root.rule is None:
        print spacing + 'Prediction:', root.prediction
        return

    print spacing + 'Rule:', str(root.rule)

    print spacing + '--> True:'
    print_tree(root.true_branch, spacing + ' ')

    print spacing + '--> False:'
    print_tree(root.false_branch, spacing + ' ')


if __name__ == '__main__':
    header = ['color', 'diameter', 'label']

    training_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 3, 'Apple'],
        ['Red', 1, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon']
    ]

    root = TreeNode(header, X=training_data)
    build_tree(root)
    print_tree(root)

    