'''
neural networks
'''
import random
import math
from numpy import dot
import numpy as np
import time 

start = time.time()
#remove this. comment and line below
random.seed(1)

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


input_data, class_labels, names = read_data("train-data.txt")
cd = {'0' : 0, '90': 1, '180':2, '270':3}


#training. input_data(read_data[0]), hidden number is number of hidden nodes. 
#output number is no of output nodes=4
def train(input_data, hidden_number, output_number=4):	
	weights_one = generate_weights(len(input_data[0]),hidden_number )
	weights_two = generate_weights(hidden_number, output_number)
	for i in range(50):
		hidden_matrix = dot_product(input_data, weights_one) #z1 = 36796*190 
		hidden_activation = [map(sigmoid, row) for row in hidden_matrix] #a1 = 36796*190 activation func applied to each element of hidden matrix z1
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

weights_one, weights_two= train(input_data, 200)
output_data, oclass_labels, names = read_data("test-data.txt")
pred = feed_forward(input_data, weights_one, weights_two)
print "Train Accuracy is" ,accuracy(pred, class_labels)
pred = feed_forward(output_data, weights_one, weights_two)
print "Test Accuracy is" ,accuracy(pred, oclass_labels)
print time.time()-start
