def is_numeric(val):
    return isinstance(val, int) or isinstance(val, float)


class SplitRule(object):
    """Rule used to partition a dataset.

    This class records the name, index and value for a particular attribute or feature of input. The
    match method of this class is used to compare whether a feature value in input vector/array
    satifies the rule. The rule is typically framed as 'is x[0] greater or equal to self.value?'
    """
    
    def __init__(self, attr_name, attr_idx, attr_val):
        self.attr_name = attr_name
        self.attr_idx = attr_idx
        self.attr_val = attr_val
    
    def match(self, x):
        """Compare feature value in a data row to the feature value in this rule.
        
        Args:
            x (list): Input of shape (D,) where D is the input dimension.
        
        Returns:
            (bool)
        """
        val = x[self.attr_idx]
        
        if is_numeric(val):
            return val >= self.attr_val
        
        return val == self.attr_val
    
    def __repr__(self):
        """Helper method to print the rule in human readable format.
        """
        condition = 'equal to'
        
        if is_numeric(self.attr_val):
            condition = 'greater or equal to'

        return "%s is %s %s" % (self.attr_name, condition, str(self.attr_val))
