'''
neural networks
'''
import random
import math
from numpy import dot
import time 

start = time.time()
#normalize data
def normalize(data):
	tr_data= zip(*data)
	norm_data =[]
	for i in tr_data:
		max_element = max(i)
		norm_data.append([float(j)/max_element for j in i])
	return zip(*norm_data)

#read data into a matrix of floats, stroes names, class_lables
#in a seperate tuple, deletes them from main data
def read_data(file_name):
	f= open(file_name, 'r')
	data = [line.strip().split() for line in f.readlines()]
	f.close()
	names = zip(*data)[0]
	class_labels = zip(*data)[1]
	for row in data:
		del row[0:2]
	return normalize([map(float,i) for i in data]),class_labels, names

#generate wieghts
def generate_weights(prev_size, next_size):
	return [[random.uniform(-1,1) for lol in range(next_size)] for lolrange in range(prev_size)]

#dot product of matrices
def dot_product(matrix_one, matrix_two):
	return dot(matrix_one, matrix_two).tolist()

#activation function
def sigmoid(x):
	return 1/(1+math.exp(-x))

#s is a sigmoid. d/dx(s) = s(1-s)
def sigmoid_derivative(s):
	return s*(1-s)

def calculate_loss(pred, act):
	return (0.5)*sum([(pred[i]-act[i])**2 for i in range(len(pred))])

def overall_loss(d, class_labels):
	loss = 0
	for i in range(len(d)):
		pred = d[i]
		act = class_labels[i]
		ind = cd[act]
		val= [0]*4
		val[ind]=1
		loss+=calculate_loss(pred, val)
	return loss

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
		


#feed forward method
def feed_forward(input_data, hidden_number, output_number):	
	weights_one = generate_weights(len(input_data[0]),hidden_number )
	weights_two = generate_weights(hidden_number, output_number)
	hidden_matrix = dot_product(input_data, weights_one)
	#print "s", len(hidden_matrix),len(hidden_matrix[0])
	hidden_activation = [map(sigmoid, row) for row in hidden_matrix]
	output_matrix = dot_product(hidden_activation, weights_two)
	pred = [map(sigmoid, row) for row in output_matrix]
 	return pred

input_data, class_labels, names = read_data("train-data.txt") 
d =  feed_forward(input_data, 10,4)
cd = {'0' : 0, '90': 1, '180':2, '270':3}

print overall_loss(d, class_labels)
print accuracy(d, class_labels)
print time.time()-start
