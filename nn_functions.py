import numpy as np
import idx2numpy
import os

class NeuralNetwork:


    def sigmoid(x):
        return 1/(1+np.exp(-x))
    
    def ReLU(x):
        return np.maximum(0.0, x)
    
    def __init__(self, design, weights=None, step_size=0.01, activation_function=sigmoid, dropout=False, bias=False):
        self.design = design
        self.step_size = step_size
        self.activation_function = activation_function # does not work properly (cannot select ReLU)
        self.bias = bias
        self.dropout = dropout
        self.weights = weights
        if self.weights is None:
            self.weights = [np.zeros(0)]
            for i in np.arange(len(self.design)-1):
                self.weights.append(np.random.uniform(-1,1,[self.design[i+1], self.design[i]]))
        self.activation = []

        if self.weights[1].shape == (self.design[1], self.design[0]):
            print('Network created successfully')
        else:
            print('Network creation failed')
    
    def train(self, input_data, target_data): # actually just place holder for batch gradient descent
        print('Training...')
        for i in np.arange(len(input_data)):
            self.one_training(input_data[i], target_data[i])
        
        if i == len(input_data) - 1:
            print('Data trained successfully')
        else:
            print('Training failed')
    
    def one_training(self, input_data, target_data):
        
        # Convert data into coumn vectors
        input_vector = np.array(input_data.flatten(), ndmin=2).T
        target_vector = np.array(target_data, ndmin=2).T
        
        # Compute activation/output
        self.activation = [] # initialize activation list
        self.activation.append(input_vector)
        for i in np.arange(len(self.design)-1):
            self.activation.append(self.activation_function(self.weights[i+1] @ self.activation[i]))
            
        # Compute error
        error = target_vector - self.activation[-1]
        
        # Update weights
        for i in np.arange(len(self.design)-1,0,-1): # move backwards through NN
            correction = self.step_size * ((error * self.activation[i] * (1.0 - self.activation[i])) @ self.activation[i-1].T)
            self.weights[i] += correction
            error = self.weights[i].T @ error
    
    def run(self, input_data):
        
        # Convert data into column vector
        input_vector = np.array(input_data.flatten(), ndmin=2).T
        
        # Compute layer outputs/activations
        self.activation = [] # initialize activation list
        self.activation.append(input_vector)
        for i in np.arange(len(self.design)-1):
            self.activation.append(self.activation_function(self.weights[i+1] @ self.activation[i]))
            
        return self.activation[-1]
    
    def evaluate(self, input_data, target_data, performance_measure=True):
        print('Testing...')
        confusion_matrix = np.zeros([target_data.shape[-1],target_data.shape[-1]]) # Initialize confusion matrix
        
        # Compute confusion matrix (one data point at a time)
        for i in np.arange(len(input_data)):
            output = self.run(input_data[i]) # get output of every data_point
            true_label = np.array(target_data[i], ndmin=2).T # get true label of every data_point
        
            # Compute result and add to confusion matrix
            confusion_result = true_label @ output.T # true label in rows, prediction in columns
            confusion_result[confusion_result == np.max(confusion_result)] = 1
            confusion_result[confusion_result != 1] = 0
            confusion_matrix += confusion_result
        
        total_predictions = np.sum(confusion_matrix)
        correct_predictions = np.sum(np.diag(confusion_matrix))
        false_predictions = total_predictions - correct_predictions
        
        # Accuracy
        self.accuracy = correct_predictions / total_predictions
        
        # Recall (per class)
        self.recall = np.array([])
        for i in np.arange(target_data.shape[-1]):
            self.recall = np.append(self.recall, confusion_matrix[i,i] / np.sum(confusion_matrix[i,:]))
        
        # Precision (per class)
        self.precision = np.array([])
        for i in np.arange(target_data.shape[-1]):
            self.precision = np.append(self.precision, confusion_matrix[i,i] / np.sum(confusion_matrix[:,i]))
        
        # Print accuracy measures
        print('Accuracy: ' + str('%.2f' % (self.accuracy * 100)) + '%')
        for i in np.arange(target_data.shape[-1]):    
            print('Recall for ' + str(i) + ': ' + str('%.2f' % (self.recall[i] * 100)) + '%')
            print('Precision for ' + str(i) + ': ' + str('%.2f' % (self.precision[i] * 100)) + '%')

    def save(self, file_name):
        np.save(file_name + '.npy', np.asarray(self.weights))
        if os.path.isfile(file_name + '.npy'):
            print('Network saved successfully as ' + file_name + '.npy')
        else:
            print('Saving failed')

# Cross validation for determining best hyperparameters
def find_hyperparameters():
    pass

# Import and pre_process data
def pre_processing():
    
    os.chdir('/Users/Max/Documents/Studium/Master/Semester_2/Programming/Data')

    # Load data and convert data to numpy arrays
    train_images = idx2numpy.convert_from_file('train-images.idx3-ubyte')
    train_labels = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
    test_images = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
    test_labels = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')

    # Re-scale input values from intervals [0,255] to [0.01,1] (necessary for optimal performance of NN)
    train_images = train_images * (0.99/255) + 0.01
    test_images = test_images * (0.99/255) + 0.01

    # Convert label data to one-hot representation with either 0.01 or 0.99 (also necessary for optimal performance of NN and to compute confusion matrix)
    train_labels = np.asfarray(train_labels) # convert to floats
    test_labels = np.asfarray(test_labels)
    train_labels = np.array(train_labels, ndmin=2).T # convert to column vector
    test_labels = np.array(test_labels, ndmin=2).T
    train_labels = (np.arange(10)==train_labels).astype(np.float) # convert to one-hot representation
    test_labels = (np.arange(10)==test_labels).astype(np.float)
    train_labels[train_labels==1] = 0.99 # substitute 0/1 for 0.01/0.99
    test_labels[test_labels==1] = 0.99
    train_labels[train_labels==0] = 0.01
    test_labels[test_labels==0] = 0.01

    if train_images.shape == (60000, 28, 28) and train_labels.shape == (60000, 10):
        print('Data preprocessed successfully')
    else:
        print('Preprocessing failed')

    os.chdir('/Users/Max/Documents/Studium/Master/Semester_2/Programming/Scripts')
    
    return train_images, train_labels, test_images, test_labels