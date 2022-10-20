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
    def __init__(self, num_input_features, num_hidden_units, num_outputs, hidden_unit_activation = SigmoidActivation, output_activation = SoftmaxActivation, loss_function=CrossEntropyLoss):
        self.num_input_features = num_input_features
        self.num_hidden_units = num_hidden_units 
        self.num_outputs = num_outputs

        self.hidden_unit_activation = hidden_unit_activation
        self.output_activation = output_activation
        self.loss_function = loss_function

        self.hidden_layer = None
        self.output_layer = None
        self.z1 = []
        self.z2 = []
        self.X = []
        self.weights = []
        self.biases = []

        self.initialize_weights()

# 64 16 10
    def initialize_weights(self):   # TODO
       
        self.weights.append(np.random.uniform(20,size=(self.num_hidden_units, self.num_input_features)))
        self.weights.append(np.random.uniform(20,size=(self.num_outputs, self.num_hidden_units)))
        self.biases.append(np.random.uniform(20,size=(self.num_hidden_units)))
        self.biases.append(np.random.uniform(20,size=(self.num_outputs)))
        # weight dimensions = (next layer, input to current layer)
        # bias dimensions = (next layer, 1)
        
        return 

    def forward(self, X, count):      # TODO
        
        if count == 0:
        
            self.X = X
            self.z1.append(([np.dot(self.weights[0], X) + self.biases[0]]))# 16x64 * 64x1 + 16x1 = 16x1 and X is flattened 8x8 image
            self.hidden_layer = self.hidden_unit_activation().__call__(np.asarray(self.z1[0]))
            self.z2 = np.asarray((np.dot(self.weights[1], self.hidden_layer) + self.biases[1]))# 10x16 * 16x1 + 10x1 = 10x1
            self.output_layer = self.output_activation().__call__(self.z2)

            return self.output_layer
        else:
            self.X = np.append(self.X,X,axis=1)
            self.z1 = np.append(self.z1,np.dot(self.weights[0], X) + self.biases[0]) # 16x64 * 64x1 + 16x1 = 16x1 and X is flattened 8x8 image
            self.hidden_layer = self.hidden_unit_activation().__call__(np.asarray(self.z1[count]))
            self.z2 = np.append((self.z2,np.dot(self.weights[1], self.hidden_layer) + self.biases[1])) # 10x16 * 16x1 + 10x1 = 10x1
            self.output_layer = self.output_activation().__call__(self.z2[count])

            return self.output_layer
        
       
        
        # x = input matrix
        # hidden activation y = f(z), where z = w.x + b
        # output = g(z'), where z' =  w'.y + b'
        # Trick here is not to think in terms of one neuron at a time
        # Rather think in terms of matrices where each 'element' represents a neuron
        # and a layer operation is carried out as a matrix operation corresponding to all neurons of the layer
    
    
    def backward(self,y_pred):     # TODO
        
        

        # Calculate the gradients of loss function w.r.t weights and biases
        # grad_cross_entropy = np.asarray()
        # grad_soft_max = np.asarray()
        # grad_sigmoid = np.asarray()
        

        grad_cross_entropy = self.loss_function().grad(y_pred) 
        grad_soft_max = self.output_activation().__grad__(self.z2) 
        del_L_del_z2 = np.dot(np.atleast_2d(grad_cross_entropy).T, np.atleast_2d(grad_soft_max)) #10X10
        del_L_del_z2 = np.sum(del_L_del_z2, axis=1) #1X10
        del_L_del_w2 = np.dot(np.atleast_2d(del_L_del_z2).T, np.atleast_2d(self.hidden_layer)) #10X16
        del_L_del_b2 = del_L_del_z2
        grad_sigmoid = self.hidden_unit_activation().__grad__(self.z1)

        del_l_del_z = np.dot(np.atleast_2d(grad_sigmoid).T,np.atleast_2d(grad_cross_entropy)) #16X10
        del_l_del_z = np.sum(del_l_del_z,axis=1) #16X1
        del_l_del_w1 = np.dot(np.atleast_2d(del_l_del_z).T, np.atleast_2d(self.X)) #16X64
        del_l_del_b1 = del_l_del_z


        return del_L_del_w2 , del_L_del_b2, del_l_del_w1, del_l_del_b1

    def update_params(self,learning_rate, del_L_del_w2 , del_L_del_b2, del_l_del_w1, del_l_del_b1):    # TODO
        self.weights[0] = self.weights[0] - learning_rate * del_l_del_w1
        self.weights[1] = self.weights[1] - learning_rate * del_L_del_w2
        self.biases[0] = self.biases[0] - learning_rate * del_l_del_b1
        self.biases[1] = self.biases[1] - learning_rate * del_L_del_b2
        return

    def train(self, dataset, learning_rate=0.0001, num_epochs=10):
       
        for epoch in range(num_epochs):
            # Forward pass
            y_pred = []
            errors = 0
            #mini_batch
            for i in range(len(dataset[0])):
                X = dataset[0][i] #10 images
                Y = dataset[1][i] #10 labels each 10 elements
                count = 0
                for x,y in zip(X,Y):
                    x = np.asarray(x)
                    y = np.asarray(y)
                  
                    x = x.flatten()
                    
                    y_pred = self.forward(x,count)
                    error = self.loss_function().__call__(y_pred, y)
                    errors = (error + errors)
                    count = count + 1
                    
                ave_error = errors/len(X) #average error for each mini batch
                
                del_L_del_w2 , del_L_del_b2, del_l_del_w1, del_l_del_b1 = self.backward(y_pred) # Backward pass
                self.update_params(learning_rate, del_L_del_w2 , del_L_del_b2, del_l_del_w1, del_l_del_b1) # Update weights and biases

                print("Epoch: {}, Loss: {}".format(epoch, np.nan_to_num(ave_error)))




    def test(self, test_dataset):
        accuracy = 0    # Test accuracy
        # Get predictions from test dataset
        # Calculate the prediction accuracy, see utils.py
        return accuracy


def main(argv):
    ann = ANN(num_input_features=64, num_hidden_units=16, num_outputs=10)

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
    for i in range(0, len(normalized_train_data), 500):
        mini_batch = (normalized_train_data[i:i+500], categorical_train_labels[i:i+500])
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
