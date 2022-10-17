import os, sys
import numpy as np
import math
import matplotlib.pyplot as plt


from data import readDataLabels, normalize_data, train_test_split, to_categorical
from utils import accuracy_score

# Create an MLP with 8 neurons
# Input -> Hidden Layer -> Output Layer -> Output
# Neuron = f(w.x + b)
# Do forward and backward propagation

mode = 'train'      # train/test... Optional mode to avoid training incase you want to load saved model and test only.

class ANN:
    def __init__(self, num_input_features, num_hidden_units, num_outputs, hidden_unit_activation, output_activation, loss_function='mse'):
        self.num_input_features = num_input_features
        self.num_hidden_units = num_hidden_units # looks like (100,100,100) for 3 hidden layers with 100 neurons each
        self.num_outputs = num_outputs

        self.hidden_unit_activation = hidden_unit_activation
        self.output_activation = output_activation
        self.loss_function = loss_function
        self.weights = []
        self.biases = []

        self.initialize_weights()

# 64 (100,100,100) 10
    def initialize_weights(self):   # TODO
       
        for i in range(1, len(self.num_hidden_units)+2):
        

            if i == 1:
                previous_layer_size = self.num_input_features
                current_layer_size = self.num_hidden_units[0]
            
            elif i == len(self.num_hidden_units)+1:
                previous_layer_size = self.num_hidden_units[-1]
                current_layer_size = self.num_outputs
            
            else:
                previous_layer_size = self.num_hidden_units[i-2]
                current_layer_size = self.num_hidden_units[i-1]

        
            weights = np.random.uniform(previous_layer_size, current_layer_size) # initialize weights randomly using a uniform distribution
            biases = np.random.uniform(1, current_layer_size) # initialize biases randomly using a uniform distribution
            self.weights.append(weights)
            self.biases.append(biases)
        
        return 

    def forward(self, X):      # TODO
        
        for i in range(self.num_hidden_units+1):
            weights = self.weights[i]
            biases = self.biases[i]

            
            Y = np.dot(weights, X) + biases
            if i == self.num_hidden_units:
                Y = self.output_activation(Y)
            else:
                Y = self.hidden_unit_activation(Y)
            return Y
        
        # x = input matrix
        # hidden activation y = f(z), where z = w.x + b
        # output = g(z'), where z' =  w'.y + b'
        # Trick here is not to think in terms of one neuron at a time
        # Rather think in terms of matrices where each 'element' represents a neuron
        # and a layer operation is carried out as a matrix operation corresponding to all neurons of the layer
    def mse(self, y_pred, y, num_samples=1):
        loss = (1/2*num_samples) * np.sum((y_pred - y)**2)
        return loss
    
    def backward(self):     # TODO
        pass
        grad_loss = []
        return grad_loss

    def update_params(self,learning_rate, grad_loss):    # TODO
        self.weights = self.weights - learning_rate * grad_loss
        return

    def train(self, dataset, learning_rate=0.01, num_epochs=10):
        for epoch in range(num_epochs):
            # Forward pass
            y_pred = []
            errors = []
            #mini_batch
            for i in range(len(dataset[0])):
                X = dataset[0][i]
                Y = dataset[1][i]
                for (x,y) in enumerate((X,Y)):
                    x = np.asarray(x)
                    y = np.asarray(y)
                    x = x.flatten()
                    y_pred = self.forward(x)

                    error = self.loss_function(y_pred, y)
                    errors.append(error)
                    # Backward pass
                    # Update weights and biases
                    grad_loss = self.backward()
                    self.update_params(learning_rate, grad_loss)
                print("Epoch: {}, Loss: {}".format(epoch, np.mean(errors)))
    def test(self, test_dataset):
        accuracy = 0    # Test accuracy
        # Get predictions from test dataset
        # Calculate the prediction accuracy, see utils.py
        return accuracy


def main(argv):
    ann = ANN()

    # Load dataset
    dataset = readDataLabels()      # dataset[0] = X, dataset[1] = y
    # Split into train and test
    train_data, train_labels, test_data, test_labels = train_test_split(dataset[0], dataset[1], n=0.8)
    # Normalize data
    normalized_train_data = normalize_data(train_data)
    normalized_test_data = normalize_data(test_data)
    # Convert labels to categorical
    categorical_train_labels = to_categorical(train_labels)
    categorical_test_labels = to_categorical(test_labels)
    #mini_batch
    X_data = []
    X_label = []
    for i in range(0, len(normalized_train_data), 100):
        mini_batch = (normalized_train_data[i:i+100], categorical_train_labels[i:i+100])
        X_data,X_label = (X_data,X_label).append(mini_batch)
    X = (X_data,X_label)
    Y = (normalized_test_data, categorical_test_labels)

    #dataset[0] = training data, dataset[1] = training labels, dataset[2] = testing data, dataset[3] = testing labels


    # call ann->train()... Once trained, try to store the model to avoid re-training everytime
    if mode == 'train':
        ann.train(X)
        ann.test(Y)        # Call ann training code here
    else:
        # Call loading of trained model here, if using this mode (Not required, provided for convenience)
        raise NotImplementedError

    # Call ann->test().. to get accuracy in test set and print it.


if __name__ == "__main__":
    main(sys.argv)
