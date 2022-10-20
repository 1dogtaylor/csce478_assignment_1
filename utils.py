import numpy as np
import math


class MSELoss:      # For Reference
    def __init__(self):
        # Buffers to store intermediate results.
        self.current_prediction = None
        self.current_gt = None
        pass

    def __call__(self, y_pred, y_gt):
        self.current_prediction = y_pred
        self.current_gt = y_gt

        # MSE = 0.5 x (GT - Prediction)^2
        loss = 0.5 * np.power((y_gt - y_pred), 2)
        return loss

    def grad(self):
        # Derived by calculating dL/dy_pred
        gradient = -1 * (self.current_gt - self.current_prediction)

        # We are creating and emptying buffers to emulate computation graphs in
        # Modern ML frameworks such as Tensorflow and Pytorch. It is not required.
        self.current_prediction = None
        self.current_gt = None

        return gradient


class CrossEntropyLoss:     # TODO: Make this work!!!
    def __init__(self):
        self.current_prediction = None
        self.current_gt = None
        pass

    def __call__(self, y_pred, y_gt):
        
        m = 10
        for i in range(m):
            self.current_prediction = y_pred[i]
            self.current_gt = y_gt[i]
            if self.current_gt == 1:
                break
                
        loss = min(10^5,(-1 * np.sum(self.current_gt * np.log(self.current_prediction))))
          
        return loss

    def grad(self, y_pred):
        gradient = y_pred - 1 # 1 = ground truth
        #wrong
        return gradient


class SoftmaxActivation:    # TODO: Make this work!!!
    def __init__(self):
        self.y = None
        pass

    def __call__(self, y):
        # TODO: Calculate Activation Function
        self.y = y
        exp_y = np.exp(y)
        sum = exp_y.sum()
        return (exp_y / sum)

    def __grad__(self, y):
        for i in range(len(y)):
            y[i] = y[i] * (1 - y[i])
        #maybe incorrect
        return y
        


class SigmoidActivation:    # TODO: Make this work!!!
    def __init__(self):
        self.y = None
        pass

    def __call__(self,y):
        # TODO: Calculate Activation Function
        self.y = y
        y = 1 / (1 + np.exp(-y))
        return y

    def __grad__(self,y):
        grad = (1/(1+np.exp(y))) * (1 - (1/(1+np.exp(y))))
        return grad


class ReLUActivation:
    def __init__(self):
        self.z = None
        pass

    def __call__(self, z):
        # y = f(z) = max(z, 0) -> Refer to the computational model of an Artificial Neuron
        self.z = z
        y = np.maximum(z, 0)
        return y

    def __grad__(self):
        # dy/dz = 1 if z was > 0 or dy/dz = 0 if z was <= 0
        gradient = np.where(self.z > 0, 1, 0)
        return gradient


def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy
