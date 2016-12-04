import collections
import time

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
    # print test_data.features
    # print len(train_lst)
    distance_vector = {}
    for train_data in train_lst:
        distance_lst = []
        for i in range(len(test_data.features)):
            # print str((test_data.features[i] - train_data.features[i])**2)
            distance_lst.append((float)(test_data.features[i] - train_data.features[i])**2)
        ed = sum(distance_lst)
        # print "ed:: " + str(ed)
        if(ed in distance_vector):
            distance_vector[ed].append(train_data)
        else:
            distance_vector[ed] = [train_data]
    return distance_vector

# What should happen when two vectors are having a same distance
def nearest_neighbors(test_lst,train_lst,k):

    for test_data in test_lst:
        distance_vector = euclidean_distance(test_data, train_lst)
        nearest_neighbors = sorted(distance_vector.keys())[:k]
        labels = []
        for n in nearest_neighbors:
            templist = distance_vector[n]
            for data in templist:
                labels.append(data.label)
        test_data.predicted = find_max_repetition(labels[0:k])

def classify(train,test,k):
    test_lst = parse_rawdata(test)
    train_lst = parse_rawdata(train)
    start_time = time.time()
    nearest_neighbors(test_lst,train_lst,k)
    for test_data in test_lst:
        print "id:: "+test_data.id+" --a:: "+test_data.label +" --p:: " +test_data.predicted
    print("--- %s seconds ---" % (time.time() - start_time))