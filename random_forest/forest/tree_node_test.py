import unittest

from tree_node import TreeNode


class TreeNodeTest(unittest.TestCase):
    def setUp(self):
        self.column_names = ['color', 'diameter', 'label']
        self.rows = [['Green', 3, 'Apple'],
                     ['Yellow', 3, 'Apple'],
                     ['Red', 1, 'Grape'],
                     ['Red', 1, 'Grape'],
                     ['Yellow', 3, 'Lemon']]
        self.node = TreeNode(self.column_names, self.rows)

    def test_prediction(self):
        """
        There are 2 apples, 2 grapes and 1 lemon
        """
        pred = self.node.prediction
        self.assertEqual(pred['Lemon'], 0.2)
        self.assertEqual(pred['Apple'], 0.4)
        self.assertEqual(pred['Grape'], 0.4)

    def test_seek_split_rule(self):
        self.assertIsNone(self.node.rule)
        self.node.seek_split_rule()
        self.assertEqual(str(self.node.rule), "diameter is greater or equal to 3")