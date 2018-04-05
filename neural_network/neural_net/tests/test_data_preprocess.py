import numpy as np
import unittest
import neuralnet.data_util as du


class DataPreprocessTest(unittest.TestCase):
    def setUp(self):
        self.data_dir = '../cifar-10-batches'
    
    def test_preprocess_cifar_10(self):
        xtr, ytr, xval, yval, xte, yte = du.preprocess_cifar_10(self.data_dir)
        self.assertEqual(xtr.shape[0], 49000)
        self.assertEqual(ytr.shape[0], 49000)
        self.assertEqual(xval.shape[0], 1000)
        self.assertEqual(yval.shape[0], 1000)
        self.assertEqual(xte.shape[0], 1000)
        self.assertEqual(yte.shape[0], 1000)        

        
    