import unittest

from split_rule import SplitRule


class SplitRuleTest(unittest.TestCase):
    def setUp(self):
        self.rule = SplitRule('age', 0, 21)
        self.true_set = [[25], [21], [30]]
        self.false_set = [[5], [10], [20]]
        
    def test_match(self):
        for x in self.true_set:
            self.assertTrue(self.rule.match(x))
        
        for x in self.false_set:
            self.assertFalse(self.rule.match(x))