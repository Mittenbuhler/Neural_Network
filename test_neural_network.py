import unittest
import numpy as np
import random
from nn_functions import *

#random.seed(123)
train_images, train_labels, test_images, test_labels = pre_processing()
neural_network = NeuralNetwork([784,10,10], bias=True)
prior_weights = neural_network.weights[1].copy()
neural_network.train(train_images[:10], train_labels[:10])
neural_network.evaluate(test_images[:100], test_labels[:100])
neural_network.save('test_network')

class Test_Neural_Network(unittest.TestCase):
    
    def test_pre_processing(self):
        self.assertTrue(train_images.shape == (60000, 28, 28) and train_labels.shape == (60000, 10))
    
    def test_init(self):
        self.assertTrue(neural_network.weights[1].shape == (neural_network.design[1], neural_network.design[0]+neural_network.bias))
    
    def test_train(self):

        self.assertTrue(np.all(prior_weights != neural_network.weights[1]))
        
    def test_run(self):
        output = neural_network.run(train_images[1])
        self.assertTrue(output.shape == (10,1))
    
    def test_evaluate(self):
        pass

    def test_save(self):
        self.assertTrue(os.path.isfile('test_network' + '.npy'))
    
if __name__ == '__main__':
    unittest.main()