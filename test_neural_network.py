import unittest
import numpy as np
from neural_network import NeuralNetwork, find_hyperparameters

class Test_Neural_Network(unittest.TestCase):
    
    def test_init(self):
        '''test if the weights are given correctly'''

        correct_weights = np.load('correct_weights.npy')

        np.random.seed(0)
        sample_network = NeuralNetwork([784,10,10])
        expected_weights = sample_network.weights[2]

        self.assertTrue(np.allclose(expected_weights, correct_weights))
    
    def test_train(self):
        '''test if the weights are trained correctly'''
        
        correct_trained_weights = np.load('correct_trained_weights.npy')

        one_image = np.load('one_image.npy')
        one_label = np.load('one_label.npy')
        np.random.seed(0)
        sample_network = NeuralNetwork([784,10,10])
        sample_network.train(one_image, one_label)
        expected_trained_weights = sample_network.weights[2]

        self.assertTrue(np.allclose(expected_trained_weights, correct_trained_weights))
        
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