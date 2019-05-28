import numpy as np
import random
import os
import unittest

from digit_recognition import NeuralNetwork, pre_processing


train_images, train_labels, test_images, test_labels = pre_processing()
neural_network = NeuralNetwork([784,100,100,10], bias=True)
prior_weights = neural_network.weights[1].copy()
neural_network.train(train_images, train_labels)
neural_network.evaluate(test_images, test_labels)
neural_network.save('test_network')


class Test_Neural_Network(unittest.TestCase):
    """ Tests each function of the NeuralNetwork class.
    
    Tests are very braod to allow improvements/modifications of the 
    neural network. They mainly ensure that the functions run without
    error and produce roughly the correct results.
    However, they also ensure that the accuracy is above
    90% which is the crucial criterion for a functioning neural network.
    """
    
    def test_pre_processing(self):
        self.assertTrue(train_images.shape == (60000, 28, 28) and 
            train_labels.shape == (60000, 10))
    
    def test_init(self):
        self.assertTrue(neural_network.weights[1].shape == (
            neural_network.design[1], 
            neural_network.design[0] + neural_network.bias,
            ))
    
    def test_train(self):
        self.assertTrue(np.all(prior_weights != neural_network.weights[1]))
        
    def test_run(self):
        output = neural_network.run(train_images[1])
        self.assertTrue(output.shape == (10,1))
    
    def test_evaluate(self):
        self.assertTrue(neural_network.confusion_matrix.sum() == 10000)
        self.assertTrue(neural_network.accuracy > 0.9)

    def test_save(self):
        self.assertTrue(os.path.isfile('test_network' + '.npy'))
    
if __name__ == '__main__':
    unittest.main()