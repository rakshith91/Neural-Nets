'''
neural networks
'''
from numpy import dot

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
	return [map(float,i) for i in data],class_labels, names

#normalize data
def normalize(data):
	return [[float(j)/max(i) for j in i] for i in zip(*data)]

#dot product of matrices
def dot_product(matrix_one, matrix_two):
	return dot(matrix_one, matrix_two).tolist()

#activation function
def sigmoid(x):
	return 1/(1+e**(-x))

def feed_forward(matrix, weights):
	z_1 = dot_product(matrix, weights) 
