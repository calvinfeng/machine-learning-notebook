from util import is_numeric


class Question(object):
    """A question is used to partition a dataset.

    This class just records a 'column number' (e.g. 0 for color) and a column value (e.g. 'green').
    The match method is used to compare the feature value in an example to the feature value stored
    in the question.
    """
    
    def __init__(self, column_name, column, value):
        self.column_name = column_name
        self.column = column
        self.value = value

    def match(self, example):
        """Compare the feature value in an example to the feature value in this question.
        """
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        
        return val == self.value

    def __repr__(self):
        """Helper method to print the question in human readable format.
        """
        condition = '=='
        if is_numeric(self.value):
            condition = '>='
        
        return "Is %s %s %s?" % (self.column_name, condition, str(self.value))
    