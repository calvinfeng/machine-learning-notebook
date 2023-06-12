import numpy as np
import unittest
import os
import neuralnet.data_util as du
 
 
class LoadCIFARTest(unittest.TestCase):
    def setUp(self):
        self.data_dir = '../cifar-10-batches'
        
    def test_load_CIFAR_batch(self):
        filepath = os.path.join(self.data_dir, 'data_batch_1')
        X, Y = du.load_CIFAR_batch(filepath)
        
        # There should be 10,000 images per batch
        self.assertEqual(X.shape[0], 10000)
        self.assertEqual(Y.shape[0], 10000)
        
        # Each image is 32 by 32
        self.assertEqual(X.shape[1], 32)
        self.assertEqual(X.shape[2], 32)
        
        # Each image has 3 channels
        self.assertEqual(X.shape[3], 3)
        
    def test_load_CIFAR10(self):
        x_training, y_training, x_test, y_test = du.load_CIFAR10(self.data_dir)
        
        # There should be 50,000 images in training set and 10,000 in test set
        self.assertEqual(x_training.shape[0], 50000)
        self.assertEqual(y_training.shape[0], 50000)
        self.assertEqual(x_test.shape[0], 10000)
        self.assertEqual(y_test.shape[0], 10000)

        # Each image is 32 by 32
        self.assertEqual(x_training.shape[1], 32)
        self.assertEqual(x_training.shape[2], 32)
        self.assertEqual(x_test.shape[1], 32)
        self.assertEqual(x_test.shape[2], 32)

        # Each image has 3 channels
        self.assertEqual(x_training.shape[3], 3)
        self.assertEqual(x_test.shape[3], 3)


