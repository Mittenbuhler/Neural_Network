import unittest
import numpy as np
from nn_functions import *

class Test_Neural_Network(unittest.TestCase):
    
    def test_init(self):
        '''test if the weights are given correctly'''

        self.assertTrue(self.weights[1].shape == (self.design[1], self.design[0]))
    
    def test_train(self):
        '''test if the weights are trained correctly'''

        self.assertTrue(i == len(input_data) - 1)
        
    def test_run(self):
        one_image = np.load('one_image.npy')
        np.random.seed(0)
        sample_network = NeuralNetwork([784,10,10])
        sample_network.run(one_image)
    
    def test_evaluate(self):
        pass
    
    def test_find_hyperparameters(self):
        pass
    
if __name__ == '__main__':
    unittest.main()