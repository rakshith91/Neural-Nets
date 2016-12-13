import knn
from nnets import *
import sys

def main():
	technique = sys.argv[3]
    # (train,test,classifier)=sys.argv[1:4]
    # (train, test, technique) = ("train-data-mod.txt","test-data-mod.txt","nearest")
    #(train, test, technique) = ("train-data.txt", "test-data.txt", "nearest")
    #define the number of neighbours to be considered
    # 1. k should be odd 2. should not be the multiple of number of classes
    #k = 200
    #print (train,test,technique)
	if technique in "nearest" :
		'''KNN is simple but complexity is the problem - if number of sample data is huge then the algorithm
        has a huge complexity'''
		knn.classify(train,test,k)
	if(technique== "nnet"):
		input_data, class_labels, names = read_data(sys.argv[1])
		cd = {'0' : 0, '90': 1, '180':2, '270':3}
		weights_one, weights_two= train(input_data,class_labels, int(sys.argv[-1]))
		output_data, oclass_labels, test_names = read_data(sys.argv[2])
		pred = feed_forward(input_data, weights_one, weights_two)
		print "Train Accuracy is" ,accuracy(pred, class_labels)
		test_pred = feed_forward(output_data, weights_one, weights_two)
		
		print "Test Accuracy is" ,accuracy(test_pred, oclass_labels)
		cf_list = write_to_file(test_pred,oclass_labels, test_names)
		print cf_list[:10]
		#print time.time()-start
	if technique == "adaboost":
		pass

main()
