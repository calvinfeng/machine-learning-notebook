def is_numeric(val):
    return isinstance(val, int) or isinstance(val, float)


def partition(rows, question):
    """Partition a dataset

    For each row in the dataset, check if it matches the question. If so, add it to the 'true rows',
    otherwise, add it to the 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
        
    return true_rows, false_rows


def class_counts(rows):
    """Counts the number of each type of example in a dataset.
    """
    counts = dict()
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        
        counts[label] += 1
    
    return counts


def gini(rows):
    """Calculate the Gini impurity for a list of rows.
    """
    counts = class_counts(rows)
    impurity = 1
    for label in counts:
        prob_of_label = counts[label] / float(len(rows))
        impurity -= prob_of_label**2
    
    return impurity


def info_gain(left, right, current_uncertainty):
    """Calculuate information gain.

    The uncertainty of the starting node, minus the weighted impurity of the two child nodes.
    """
    prob = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - prob * gini(left) - (1-prob) * gini(right)