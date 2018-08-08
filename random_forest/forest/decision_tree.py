from tree_node import TreeNode


class DecisionTree(object):
    def __init__(self):
        self.root = None

    def fit(self, header, training_data):
        self.root = TreeNode(header, X=training_data)

        def build_tree(node):
            if node.split() is not None:
                build_tree(node.true_branch)
                build_tree(node.false_branch)
        
        build_tree(self.root)

    def display(self):
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
        def _classify(node, x):
            if node.rule is None:
                return node.prediction
            
            if node.rule.match(x):
                return _classify(node.true_branch, x)
            
            return _classify(node.false_branch, x)
        
        return _classify(self.root, x)


if __name__ == '__main__':
    header = ['color', 'diameter', 'label']

    training_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 3, 'Apple'],
        ['Red', 1, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon']
    ]

    tree = DecisionTree()
    tree.fit(header, training_data)
    tree.display()
    print tree.predict(['Red', 4])