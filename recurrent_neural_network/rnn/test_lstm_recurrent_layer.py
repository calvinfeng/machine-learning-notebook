# Created: April, 2018
# Author(s): Calvin Feng

import numpy as np
import unittest
from lstm_recurrent_layer import LSTMRecurrentLayer
from gradient_check import eval_numerical_gradient_array, rel_error


class LSTMRecurrentLayerTest(unittest.TestCase):
    def test_forward_step(self):
        N, D, H = 3, 4, 5

        # Create the layer
        layer = LSTMRecurrentLayer(D, H)
        layer.Wx = np.linspace(-2.1, 1.3, num=4*D*H).reshape(D, 4*H)
        layer.Wh = np.linspace(-0.7, 2.2, num=4*H*H).reshape(H, 4*H)
        layer.b = np.linspace(0.3, 0.7, num=4*H)

        # Create some arbitrary inputs
        x = np.linspace(-0.4, 1.2, num=N*D).reshape(N, D)
        prev_h = np.linspace(-0.3, 0.7, num=N*H).reshape(N, H)
        prev_c = np.linspace(-0.4, 0.9, num=N*H).reshape(N, H)

        next_h, next_c, _ = layer._forward_step(x, prev_h, prev_c)

        expected = np.asarray([
            [ 0.24635157,  0.28610883,  0.32240467,  0.35525807,  0.38474904],
            [ 0.49223563,  0.55611431,  0.61507696,  0.66844003,  0.7159181 ],
            [ 0.56735664,  0.66310127,  0.74419266,  0.80889665,  0.858299  ]])
        self.assertAlmostEqual(rel_error(expected, next_h), 1e-9, places=2)

        expected = np.asarray([
            [ 0.32986176,  0.39145139,  0.451556,    0.51014116,  0.56717407],
            [ 0.66382255,  0.76674007,  0.87195994,  0.97902709,  1.08751345],
            [ 0.74192008,  0.90592151,  1.07717006,  1.25120233,  1.42395676]])
        self.assertAlmostEqual(rel_error(expected, next_c), 1e-9, places=2)

    def test_forward(self):
        N, D, H, T = 2, 5, 4, 3

        # Create the layer
        layer = LSTMRecurrentLayer(D, H)
        layer.Wx = np.linspace(-0.2, 0.9, num=4*D*H).reshape(D, 4*H)
        layer.Wh = np.linspace(-0.3, 0.6, num=4*H*H).reshape(H, 4*H)
        layer.b = np.linspace(0.2, 0.7, num=4*H)

        # Create some arbitrary inputs
        x = np.linspace(-0.4, 0.6, num=N*T*D).reshape(N, T, D)
        h0 = np.linspace(-0.4, 0.8, num=N*H).reshape(N, H)

        hidden_state_over_time = layer.forward(x, h0)

        expected = np.asarray([
            [[ 0.01764008,  0.01823233,  0.01882671,  0.0194232 ],
             [ 0.11287491,  0.12146228,  0.13018446,  0.13902939],
             [ 0.31358768,  0.33338627,  0.35304453,  0.37250975]],
            [[ 0.45767879,  0.4761092,   0.4936887,   0.51041945],
             [ 0.6704845,   0.69350089,  0.71486014,  0.7346449 ],
             [ 0.81733511,  0.83677871,  0.85403753,  0.86935314]]
        ])    

        self.assertAlmostEqual(rel_error(hidden_state_over_time, expected), 1e-9, places=2)
    
    def test_backward(self):
        np.random.seed(271)

        N, D, T, H = 2, 3, 10, 6
        
        # Create the layer
        layer = LSTMRecurrentLayer(D, H)        
        Wx = np.random.randn(D, 4*H)
        Wh = np.random.randn(H, 4*H)
        b = np.random.randn(4*H)

        # Create some arbitrary inputs
        x = np.random.randn(N, T, D)
        h0 = np.random.randn(N, H)

        hidden_state_over_time = layer.forward(x, h0, Wx=Wx, Wh=Wh, b=b)

        grad_h_over_time = np.random.randn(*hidden_state_over_time.shape)

        grad_x, grad_h0, grad_Wx, grad_Wh, grad_b = layer.backward(grad_h_over_time)

        fx = lambda x: layer.forward(x, h0, Wx=Wx, Wh=Wh, b=b)
        fh0 = lambda h0: layer.forward(x, h0, Wx=Wx, Wh=Wh, b=b)
        fWx = lambda Wx: layer.forward(x, h0, Wx=Wx, Wh=Wh, b=b)
        fWh = lambda Wh: layer.forward(x, h0, Wx=Wx, Wh=Wh, b=b)
        fb = lambda b: layer.forward(x, h0,  Wx=Wx, Wh=Wh, b=b)

        grad_x_num = eval_numerical_gradient_array(fx, x, grad_h_over_time)
        grad_h0_num = eval_numerical_gradient_array(fh0, h0, grad_h_over_time)
        grad_Wx_num = eval_numerical_gradient_array(fWx, Wx, grad_h_over_time)
        grad_Wh_num = eval_numerical_gradient_array(fWh, Wh, grad_h_over_time)
        grad_b_num = eval_numerical_gradient_array(fb, b, grad_h_over_time)

        self.assertAlmostEqual(rel_error(grad_x_num, grad_x), 1e-9, places=2)
        self.assertAlmostEqual(rel_error(grad_h0_num, grad_h0), 1e-9, places=2)
        self.assertAlmostEqual(rel_error(grad_Wx_num, grad_Wx), 1e-9, places=2)
        self.assertAlmostEqual(rel_error(grad_Wh_num, grad_Wh), 1e-9, places=2)
        self.assertAlmostEqual(rel_error(grad_b_num, grad_b), 1e-9, places=2)

