import numpy as np


def partition(X, rule):
    """Partition the inputs

    For each row in the input, check if it satifies the rule. If so, add it to the true set,
    otherwise, add it to the false set.

    Args:
        X (2D list): A list of inputs, where each input is a list of feature values.
        rule (SpltRule): Rule for performing partitioning.
    """
    true_set, false_set = [], []
    for x in X:
        if rule.match(x):
            true_set.append(x)
        else:
            false_set.append(x)
        
    return true_set, false_set


def class_counts(X):
    """Counts the number of each type of example in a dataset.
    """
    counts = dict()
    for x in X:
        label = x[-1]
        if label not in counts:
            counts[label] = 0
        
        counts[label] += 1
    
    return counts


def gini(X):
    """Calculate the Gini impurity for a list of inputs.
    """
    counts = class_counts(X)
    impurity = 1
    for label in counts:
        prob_of_label = counts[label] / float(len(X))
        impurity -= prob_of_label**2
    
    return impurity
    

def purity_gain(partitions, curr_impurity, N):
    """Calculate the reduction in impurity after partitions
    """
    gain = curr_impurity
    for part in partitions:
        prob = float(len(part)) / N
        gain -= prob * gini(part)
    
    return gain


def entropy(X):
    """Compute entropy for a list of inputs, based on their labels.
    """
    counts = class_counts(X)
    entropy = 0
    for label in counts:
        prob = counts[label] / float(len(X))
        entropy += -1 * prob * np.log(prob)
    
    return entropy


def info_gain(partitions, curr_entropy, N):
    """Compute information gain by comparing the difference of entropies before and after partition.
    """    
    gain = curr_entropy
    for part in partitions:
        prob = float(len(part)) / N
        gain -= prob * entropy(part)
        
    return gain