from tree_node import TreeNode


class DecisionTree(object):
    def __init__(self, feat_indices=None):
        self.root = None
        self.feat_indices = feat_indices

    def fit(self, column_names, training_data):
        """Construct a decision tree
        """
        self.root = TreeNode(column_names, X=training_data, feat_indices=self.feat_indices)

        def build_tree(node):
            if node.split() is not None:
                build_tree(node.true_branch)
                build_tree(node.false_branch)
        
        build_tree(self.root)

    def display(self):
        """Prints a decision tree to screen
        """
        def _print_tree(node, spacing=''):
            if node.rule is None:
                print spacing + 'Prediction:', node.prediction
                return

            print spacing + 'Rule:', str(node.rule)

            print spacing + '--> True:'
            _print_tree(node.true_branch, spacing + ' ')

            print spacing + '--> False:'
            _print_tree(node.false_branch, spacing + ' ')    

        if self.root is not None:
            _print_tree(self.root)

    def predict(self, x):
        """Makes a prediction using the provided data row
        """
        def _classify(node, x):
            if node.rule is None:
                return node.prediction
            
            if node.rule.match(x):
                return _classify(node.true_branch, x)
            
            return _classify(node.false_branch, x)
        
        return _classify(self.root, x)


if __name__ == '__main__':
    column_names = ['color', 'diameter', 'label']

    training_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 3, 'Apple'],
        ['Red', 1, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon']
    ]

    tree = DecisionTree()
    tree.fit(column_names, training_data)
    tree.display()
    print tree.predict(['Red', 4])