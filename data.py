import os, sys
import numpy as np
from sklearn import datasets

#import libraries as needed

def readDataLabels(): 
	#read in the data and the labels to feed into the ANN
	data = datasets.load_digits()
	X = data.data
	y = data.target

	return X,y

def to_categorical(y):
	# 0 -> [1,0,0,0,0,0,0,0,0,0]
	# 7 -> [0,0,0,0,0,0,0,1,0,0]
	# 2 -> [0,0,1,0,0,0,0,0,0,0]
	#Convert the nominal y values to categorical
	y = np.array(y)
	n_values = np.max(y) + 1
	y = np.eye(n_values)[y]

	return y
	
def train_test_split(data,labels,n=0.8): #TODO

	mask = np.random.rand(len(data)) < n
	train_data = data[mask]
	train_labels = labels[mask]
	test_data = data[~mask]
	test_labels = labels[~mask]
	#split data in training and testing sets
	splitdata = [train_data,train_labels,test_data,test_labels] #dataset[0] = training data, dataset[1] = training labels, dataset[2] = testing data, dataset[3] = testing labels
	return splitdata

def normalize_data(data): #TODO

	x = data
	x = x.astype('float32')
	x /= 255
	# normalize/standardize the data

	return x
