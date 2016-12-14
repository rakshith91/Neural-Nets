import collections
import numpy as np
import time

'''

Parameters
K:              Indicates the number of nearest neighbours to be considered. Can be changed at Runtime
                    1. k should be odd
                    2. should not be the multiple of number of classes
feature_count:  Its hard coded in the program but can be adjusted by changing the "matrix" variable in filter() in knn.py
Accuracy:       Accuracy of the classifier, comparing with the predicted and actual values

K   feature_count(8x8 = 64) Accuracy    Time(On Local Machine)
200        20               71.36       680 sec
100        20               71.89       680 sec
101        12               70.6        442 sec
53         20               70.51       650 sec


How code Works:
For each test_data
1. Euclidean distance of the test_data from every training data is calculated
2. Then the K nearest neighbours with respect to the distance is considered
3. Among the K nearest neighbours labels having higher count will be the predicted label of the test data

Problems faced
Choosing the correct features -- solution is using a filter as mentioned below
Choosing the correct distance vector for calculating distance between two points : We came up with an idea where we consider only
select pixels across two images to calculate the distance.

Simplification --Intuition of choosing edges in the images
Filter:
This filter function considers only the pixels configured in the matrix variable
0 degree - images will have lighter pixels at the top and darker pixels at bottom this implies
90 degree - image is rotated lighter pixels at the right and darker pixed in the left
180 degree - image is rotated lighter pixels at the bottom and darker pixed in the top
270 degree - image is rotated lighter pixels at the left and darker pixed in the right


'''
def filter():
    max_col = 192
    tup_size = 3

    matrix = [[1, 1, 1, 0, 0, 1, 1, 1], \
              [1, 0, 0, 0, 0, 0, 0, 1], \
              [1, 0, 0, 0, 0, 0, 0, 1], \
              [0, 0, 0, 0, 0, 0, 0, 0], \
              [0, 0, 0, 0, 0, 0, 0, 0], \
              [1, 0, 0, 0, 0, 0, 0, 1], \
              [1, 0, 0, 0, 0, 0, 0, 1], \
              [1, 1, 1, 0, 0, 1, 1, 1]]

    # matrix = [[1, 1, 0, 0, 0, 0, 1, 1], [1, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0,0, 1, 1]]
    f = np.array(matrix).ravel()
    feature_idx = []
    for i in range(0, max_col):
        idx = (i + tup_size) / tup_size
        if (f[idx - 1] == 1):
            # print str(idx) +":: "+ str(i)
            feature_idx.append(i)
    return feature_idx

class Data(object):
    def __init__(self,id,label,features):
        self.id = id
        self.label = label
        self.features = features
        self.predicted = ''

    def print_data(self):
        print self.id
        print self.label
        print self.features

    def __str__(self):
        return str(self.id) +";"+str(self.label)

def find_max_repetition(lst):
    # Returns the tuple having max
    counter = collections.Counter(lst)
    max = ('',-1)
    for element in counter.items():
        if(element[1] > max[1]):
            max = element
    return max[0]

def parse_rawdata(filename):
    file = open(filename)
    data_lst = list()
    for line in file:
        chunk = line.strip().split(" ")
        data_lst.append(Data(chunk[0],chunk[1],map(int, chunk[2:])))
    return data_lst

#Calculates the euclidean distance between a single test data and  the list of testing data
def euclidean_distance(test_data,train_lst):
    print "IN:: euclidean_distance"
    # distance_vector is map so as to handle duplicate distance
    distance_vector = {}
    for train_data in train_lst:
        distance_lst = []
        for i in range(len(test_data.features)):
            distance_lst.append((float)(test_data.features[i] - train_data.features[i])**2)
        ed = sum(distance_lst)
        if(ed in distance_vector):
            distance_vector[ed].append(train_data)
        else:
            distance_vector[ed] = [train_data]
    return distance_vector

# Calculates the Euclidean Distance between the testdata and all the training data
def euclidean_distance_i(test_data,train_lst,feature_lst):
    # distance_vector is map<distance,list of data points> so as to handle data points having same distance from test data
    distance_vector = {}
    for train_data in train_lst:
        distance_lst = []
        for i in feature_lst:
            distance_lst.append((test_data.features[i] - train_data.features[i])**2)
        ed = sum(distance_lst)
        if(ed in distance_vector):
            distance_vector[ed].append(train_data)
        else:
            distance_vector[ed] = [train_data]
    return distance_vector

# What should happen when two vectors are having a same distance
def nearest_neighbors(test_lst,train_lst,k):
    filter_idx = filter()

    for test_data in test_lst:
        distance_vector = euclidean_distance_i(test_data, train_lst,filter_idx)
        # sorted(distance_vector.keys())[:k] - gets the k nearest values from the distance_vector
        nearest_neighbors = sorted(distance_vector.keys())[:k]
        labels = []
        # Extracts the labels from K neighbours
        for n in nearest_neighbors:
            templist = distance_vector[n]
            for data in templist:
                labels.append(data.label)

        test_data.predicted = find_max_repetition(labels[0:k])
        # break

def classify(train,test,k):
    test_lst = parse_rawdata(test)
    train_lst = parse_rawdata(train)
    start_time = time.time()
    print("--- start ---" , (start_time))
    nearest_neighbors(test_lst,train_lst,k)
    i = 0
    correct = 0
    result = []
    f = open("knn_output.txt", "w")
    f.close()
    print "K = "+str(k)
    for test_data in test_lst:
        # print str(i)+" --id:: "+test_data.id+" --a:: "+test_data.label +" --p:: " +test_data.predicted
        if(test_data.label == test_data.predicted):
            correct+=1
        i+=1
        result.append((test_data.label,test_data.predicted))
        f = open("knn_output.txt", "a")
        f.write(test_data.id + " " + str(test_data.predicted) + "\n")
        f.close()
    print "Accuracy:: ",float(correct)*100/(i),"%"
    print("--- %s seconds ---" % (time.time() - start_time))
    return result
