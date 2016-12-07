'''
neural networks
'''
import random
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
	return norm_data

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
def generate_weights(prev_size, nex_size):
	return [[random.uniform(-1,1) for lol in range(next_size)] for lolrange in range(prev_size)]

#dot product of matrices
def dot_product(matrix_one, matrix_two):
	return dot(matrix_one, matrix_two).tolist()

#activation function
def sigmoid(x):
	return 1/(1+e**(-x))

#feed forward method
def feed_forward(input_data, weights_one, weights_two):
	
	hidden_matrix = dot_product(matrix, weights_one)
	hidden_activation = [map(sigmoid, row) for row in hidden_matrix]
	output_matrix = dot_product(hidden_activation, weights_two)
	pred = [map(sigmoid, row) for row in output_matrix]
 	return pred
 
