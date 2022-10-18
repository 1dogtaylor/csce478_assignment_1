import os, sys
import numpy as np
import math
import matplotlib.pyplot as plt


from data import readDataLabels, normalize_data, train_test_split, to_categorical
from utils import accuracy_score, MSELoss, CrossEntropyLoss, ReLUActivation, SigmoidActivation, SoftmaxActivation
# Create an MLP with 8 neurons
# Input -> Hidden Layer -> Output Layer -> Output
# Neuron = f(w.x + b)
# Do forward and backward propagation

mode = 'train'      # train/test... Optional mode to avoid training incase you want to load saved model and test only.

class ANN:
    def __init__(self, num_input_features, num_hidden_units, num_outputs, hidden_unit_activation = ReLUActivation, output_activation = SoftmaxActivation, loss_function=MSELoss):
        self.num_input_features = num_input_features
        self.num_hidden_units = num_hidden_units 
        self.num_outputs = num_outputs

        self.hidden_unit_activation = hidden_unit_activation
        self.output_activation = output_activation
        self.loss_function = loss_function

        self.hidden_layer = None
        self.output_layer = None
        self.weights = []
        self.biases = []

        self.initialize_weights()

# 64 16 10
    def initialize_weights(self):   # TODO
       
        self.weights.append(np.random.uniform(self.num_hidden_units, self.num_input_features))
        self.weights.append(np.random.uniform(self.num_outputs, self.num_hidden_units))
        self.biases.append(np.random.uniform(self.num_hidden_units, 1))
        self.biases.append(np.random.uniform(self.num_outputs, 1))
        # weight dimensions = (next layer, input to current layer)
        # bias dimensions = (next layer, 1)
        
        return 

    def forward(self, X):      # TODO
        self.z1 = np.dot(self.weights[0], X) + self.biases[0] # 16x64 * 64x1 + 16x1 = 16x1 and X is flattened 8x8 image
        self.hidden_layer = self.hidden_unit_activation(self.z1)
        self.z2 = np.dot(self.weights[1], self.hidden_layer) + self.biases[1] # 10x16 * 16x1 + 10x1 = 10x1
        self.output_layer = self.output_activation(self.z2)

        return self.output_layer
        
       
        
        # x = input matrix
        # hidden activation y = f(z), where z = w.x + b
        # output = g(z'), where z' =  w'.y + b'
        # Trick here is not to think in terms of one neuron at a time
        # Rather think in terms of matrices where each 'element' represents a neuron
        # and a layer operation is carried out as a matrix operation corresponding to all neurons of the layer
    
    
    def backward(self,errors):     # TODO
        
        
        pass
        grad_cross_entropy = np.asarray() # dl/dy_pred

        #put in the softmax derivative to give us grad_z1
        grad_z2 = np.asarray() # dy_pred/dz2

        #put in the sigmoid derivative this gives us grad z
        grad_z1 = np.asarray() # dz2/dz1

        grad_z = np.asarray() # dz1/dz
        

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

                    error = self.loss_function.__call__(y_pred, y)
                    errors = np.sum(error,errors)
                    # Backward pass
                    # Update weights and biases
                    grad_loss = self.backward(errors)
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
        X_data.append(mini_batch[0])
        X_label.append(mini_batch[1])
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
