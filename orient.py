import knn
from nnets import *
import nnets
import sys
import adaboost

"""
Please refer to the start of following files for more description
1. adaboost.py
2. knn.py
3. nnet.py
In assignment KNN give better accuracy (71.3 %)but takes 10 min for testing 943 values

"""

def confusionMatrix(cm):
    lst = (0, 90, 180, 270)
    matrix = [[0, 0, 0, 0], \
              [0, 0, 0, 0], \
              [0, 0, 0, 0], \
              [0, 0, 0, 0]]

    for actual, pred in cm:
        i = lst.index(int(actual))
        j = lst.index(int(pred))
        matrix[i][j] += 1

    print "Confusion Matrix is Below: (0,90,180,270)"
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            print str(matrix[i][j]) + " ",
        print "\n"


def main():
    technique = sys.argv[3]

    if (technique in ("nearest","best")):
        if technique=="best":
            print "We are getting best accuracy for KNN, and adaboost is closely behind it..."
        k = 201
        print "inside KNN , this takes upto to 15  minutes to show the output"
        train = sys.argv[1]
        test = sys.argv[2]
        # (train, test, technique) = ("train-data-mod.txt", "test-data-mod.txt", "nearest")
        '''KNN is simple but complexity is the problem - if number of sample data is huge then the algorithm
        has a huge complexity'''
        result = knn.classify(train, test, k)
        confusionMatrix(result)
        print "knn_output.txt is created. You can see our predictions there"
    if (technique == "nnet"):
        input_data, class_labels, names = read_data(sys.argv[1])
        cd = {'0': 0, '90': 1, '180': 2, '270': 3}
        weights_one, weights_two = nnets.train(input_data, class_labels, int(sys.argv[-1]))
        output_data, oclass_labels, test_names = read_data(sys.argv[2])
        pred = feed_forward(input_data, weights_one, weights_two)
        print "Train Accuracy is", accuracy(pred, class_labels)
        test_pred = feed_forward(output_data, weights_one, weights_two)

        print "Test Accuracy is", accuracy(test_pred, oclass_labels)
        cf_list = write_to_file(test_pred, oclass_labels, test_names)
        confusionMatrix(cf_list)
        print "nnet_output.txt is created. You can see our predictions there"
        #print cf_list[:10]
    # print time.time()-start
    if technique == "adaboost":
        print "hii... this would take around 1 minute to run"
        adaboost.main(sys.argv[1], sys.argv[2], int(sys.argv[4]))
        print "adaboost_output.txt is created. You can see our predictions there"

main()
