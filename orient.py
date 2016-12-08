import knn

import sys

def main():
    # (train,test,classifier)=sys.argv[1:4]
    # (train, test, technique) = ("train-data-mod.txt","test-data-mod.txt","nearest")
    (train, test, technique) = ("train-data.txt", "test-data.txt", "nearest")
    #define the number of neighbours to be considered
    # 1. k should be odd 2. should not be the multiple of number of classes
    k = 200
    print (train,test,technique)
    if(technique == "nearest"):
        '''KNN is simple but complexity is the problem - if number of sample data is huge then the algorithm
        has a huge complexity'''
        knn.classify(train,test,k)

main()