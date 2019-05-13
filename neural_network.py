import numpy as np


class NeuralNetwork:


    def sigmoid(x):
        return 1/(1+np.exp(-x))
    
    def ReLU(x):
        return np.maximum(0.0, x)
    
    def __init__(self, design, step_size=0.01, activation_function=sigmoid, dropout=False, bias=False):
        self.design = design
        self.step_size = step_size
        self.activation_function = activation_function # does not work properly (cannot select ReLU)
        self.bias = bias
        self.dropout = dropout
        self.create_weights()
        self.activation = []
    
    def create_weights(self):
        self.weights = [np.zeros(0)]
        for i in np.arange(len(self.design)-1):
            self.weights.append(np.random.uniform(-1,1,[self.design[i+1], self.design[i]]))
    
    def train(self, input_data, target_data): # actually just place holder for batch gradient descent
        for i in np.arange(len(input_data)):
            self.one_training(input_data[i], target_data[i])
    
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


# Cross validation for determining best hyperparameters
def find_hyperparameters():
    pass