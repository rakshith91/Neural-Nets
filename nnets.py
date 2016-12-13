'''
neural networks

The program takes about 6 mins for 25 iterations. 

1) Problem description

The problem is to classify image orientation using neural networks. It involves two steps:feed forward and back propagation.

There is an input layer with 192 nodes in it, hidden layer with a user specified number of nodes and 4 output nodes in the output layer.

Each node in the previous layer is connected to all the nodes in the next layer forming a trellis. Each edge has a weight.

The weight matrix from input layer to hidden layer is called weights_one
The weight matrix from hidden layer to output layer is called weights_two

In the feed forward, the weights are generate randomly from a uniform distribution between (-1, 1).

Before running the feed forward, the data is normalised . Every value x in column C is normalised to x/max(C).

Once the hidden nodes values are computed , a sigmoid activation function is used. 

The train functions does feed forward once and backpropagation 15 times and keeps learning the weights. 

Every time, step function of 0.1 is used along with a regularization parameter of 0.01

The loss function used is 1/2(y-y_hat)**2.This shows the overall loss in training the data



2)description of program 

The transpose function returns transpose of a matrix.

normalize returns the normalized matrix.

read_data returns a tuple of data, features and image names

generate_weights generates the weight matrix 

loss_matrix compute the loss after every computation.

feed_forward function is to perform feed forward with given weights.i.e. to test the data.

finally train returns the weights tuple after training on the train data and write_to_file writes the output.

3)discussion of any problems

There are several problems faced :

i) decision of the activation function between sigmoid, tanh : sigmoid has been chosen

ii)Due to high bias, all the examples are getting classified to either 90 or 180.



'''
import random
import math
from numpy import dot
import numpy as np
import time 
import sys
#start = time.time()
#remove this. comment and line below
#random.seed(1)

#transpose matrix function. x is a matrix
def transpose(x):
	return zip(*x)

#normalize data 
def normalize(data):
	tr_data= transpose(data)
	norm_data =[]
	for i in tr_data:
		max_element = max(i)
		norm_data.append([float(j)/max_element for j in i])
	return transpose(norm_data)

#read data into a matrix of floats, stroes names, class_lables
#in a seperate tuple, deletes them from main data
def read_data(file_name):
	f= open(file_name, 'r')
	data = [line.strip().split() for line in f.readlines()]
	f.close()
	names = transpose(data)[0]
	class_labels = transpose(data)[1]
	for row in data:
		del row[0:2]
	return normalize([map(float,i) for i in data]),class_labels, names


#generate weights randomly.prev_size is the no of nodes in previous layer
#next_size is the no of nodes in next_layer
def generate_weights(prev_size, next_size):
	return [[random.uniform(-1,1) for i in range(next_size)] for j in range(prev_size)]


#dot product of matrices 
def dot_product(matrix_one, matrix_two):
	return dot(matrix_one, matrix_two).tolist()

#activation function
def sigmoid(x):
	return 1/(1+math.exp(-x))

#s is a sigmoid. d/dx(s) = s(1-s)
def sigmoid_derivative(s):
	return s*(1-s)

#calculates the loss matrix.there are 4 output nodes.loss matrix size is 36976*4
def loss_matrix(pred, act):
	loss_matrix=[]
	cd = {'0' : 0, '90': 1, '180':2, '270':3}
	for i in range(len(act)):
		ind = cd[act[i]]
		val=[0]*4
		val[ind]=1
		loss_matrix.append([0.5*((pred[i][j]-val[j])**2) for j in range(len(val))])
	return loss_matrix

#calculates loss for a single row
def calculate_loss(pred, act):
	return (0.5)*sum([(pred[i]-act[i])**2 for i in range(len(pred))])

#sum of all losses. d is the prediction matrix of size 36976*4
def overall_loss(d, class_labels):
	cd = {'0' : 0, '90': 1, '180':2, '270':3}
	loss = 0.0
	for i in range(len(d)):
		pred = d[i]
		act = class_labels[i]
		ind = cd[act]
		val= [0]*4
		val[ind]=1
		loss+=calculate_loss(pred, val)
	return loss/len(d)

#accuracy for prediction, d is the prediction matrix of size 36976*4
def accuracy(d, class_labels):
	cd = {'0' : 0, '90': 1, '180':2, '270':3}
	count = 0 	
	for i in range(len(d)):
		pred = d[i]
		pred_val = pred.index(max(pred))
		act = class_labels[i]
		ind = cd[act]
		if pred_val == ind:
			count+=1.0
	return count/len(class_labels)
		


#predict the method. used after training. data should be test data. i.e. read_data[0]
#weights_one , weights_two are weight matrices. 
def feed_forward(data, weights_one, weights_two):	
	hidden_matrix = dot_product(data, weights_one)
	hidden_activation = [map(sigmoid, row) for row in hidden_matrix]
	output_matrix = dot_product(hidden_activation, weights_two)
	pred = [map(sigmoid, row) for row in output_matrix]
 	return pred

#training. input_data(read_data[0]), hidden number is number of hidden nodes. 
#output number is no of output nodes=4
def train(input_data, class_labels,hidden_number, output_number=4):	
	weights_one = generate_weights(len(input_data[0]),hidden_number )
	weights_two = generate_weights(hidden_number, output_number)
	for i in range(15):
		hidden_matrix = dot_product(input_data, weights_one) #z1 = 36796*hidden nodes
		hidden_activation = [map(sigmoid, row) for row in hidden_matrix] #a1 = 36796*hidden nodes activation func applied to each element of hidden matrix z1
		output_matrix = dot_product(hidden_activation, weights_two) #using the generate a1(hideen_activation), find the values at output nodes for each record. outputs 36976*4 matrix each row represents probabilty of the record belonging to eachn node.
		#prediction will be element wise sigmoid for each element in output matrix
		pred = [map(sigmoid, row) for row in output_matrix]
		#calculate the loss matrix. using actual class_labels
		delta  = loss_matrix(pred, class_labels)
		#using delta, we find dw2 .i.e try to learn weights
		deriv_W2 = dot_product(transpose(hidden_activation) , delta)
		#delta 2 .i.e loss for input and hidden
		delta2 = np.multiply(np.array(dot_product(delta, transpose(weights_two))),(np.array([map(sigmoid_derivative, row) for row in hidden_activation])) ).tolist()
		#dw1
		deriv_W1 = dot_product(transpose(input_data), delta2)

		#some regularization using step size of 0.01
		deriv_W2  = [map(lambda i:i+(-0.01*i) , row) for row in weights_two]
		deriv_W1 = [map(lambda i:i+(-0.01*i) , row) for row in weights_one]
		weights_one = [map(lambda i:i+(-0.1*i) , row) for row in deriv_W1]
		weights_two =[map(lambda i:i+(-0.1*i) , row) for row in deriv_W2]
		
		loss = overall_loss(pred, class_labels)
		print "loss",loss
	return weights_one, weights_two 

#writes output to a file
def write_to_file(prediction,actual, image_names):
	cd = {'0' : 0, '90': 1, '180':2, '270':3}
	orient = {0 : 0, 1: 90, 2: 180, 3 : 270}
	f = open("nnet_output.txt", "w")
	cf_list=[]
	for i in range(len(prediction)):
		row = prediction[i]
		ind = row.index(max(row))
		cf_list.append((actual[i], orient[ind]))
		f.write(image_names[i]+" "+str(orient[ind])+"\n")
	f.close()
	return cf_list
