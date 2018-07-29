from util import partition, gini, class_counts, info_gain
from question import Question
import numpy as np


class Leaf(object):
    """Leaf node that classifies data.

    This holds a dictionary of class (e.g. 'Apple') mapping to the number of times it appears in the 
    rows from the training data that reach this leaf.
    """
    def __init__(self, rows):
        self.class_counts = class_counts(rows)

    @property
    def predictions(self):
        norm = np.sum(self.class_counts.values())
        pred = self.class_counts.copy()
        for key in pred:
            pred[key] = float(pred[key])/norm
        
        return pred


class DecisionNode(object):
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def find_best_split(col_names, rows):
    """Find the best question to ask by iterating over every feature / value and calculating the information gain.
    """
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    num_feats = len(rows[0]) - 1

    for col in range(num_feats):
        values = set([row[col] for row in rows])  # Extract unique values from the dataset in a given column.
        for val in values:
            # Create a question, if the value is numeric, then the question is whether input is greater or equal to val.
            # If the value is string, then the question si whether input is equal to val.
            question = Question(col_names[col], col, val)

            # Split the dataset
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            
            # Calculate information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question
    
    return best_gain, best_question


def build_tree(col_names, rows):
    """Build a decision tree recursively.

    The function will return a top level decision node which has multiple branches if the rows are separable. One can 
    use the root node to traverse through the tree and seek leaf nodes.
    """
    gain, question = find_best_split(col_names, rows)

    if gain == 0:
        return Leaf(rows)
    
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(col_names, true_rows)
    false_branch = build_tree(col_names, false_rows)

    return DecisionNode(question, true_branch, false_branch)


def print_tree(node, spacing=''):
    if isinstance(node, Leaf):
        print spacing + 'Predict', node.predictions
        return
    
    print spacing + str(node.question)

    print spacing + '--> True:'
    print_tree(node.true_branch, spacing + '  ')
    
    print spacing + '--> False:'
    print_tree(node.false_branch, spacing + '  ')


def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    
    if node.question.match(row):
        return classify(row, node.true_branch)
    
    return classify(row, node.false_branch)
    

if __name__ == '__main__':
    header = ['color', 'diameter', 'label']

    training_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 3, 'Apple'],
        ['Red', 1, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon']
    ]

    #####################
    # Construct the tree 
    #####################
    root = build_tree(header, training_data)
    print_tree(root)

    testing_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 4, 'Apple'],
        ['Red', 2, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon']
    ]

    for row in testing_data:
        print 'Actual label is %s and predicted %s' % (row[-1], classify(row, root))