import math
import random
import pickle
import copy
import time
"""
We have implemented the adaboost technique in this file.

1) Problem description:
  The problem is to classify image orientation using adaboost. We have some changeable paramters like number of stumps,
  types of stumps, type of decission at every decission node.

2) Program logic:
   Number of stumps is an input for this program. For every stump we are actually creating 4 stumps, one each for
   0 classifier , 90 classifier, 180 classifier, 270 classifier
   We are selecting 'n' number of random pairs initially. Every pair has two pixel values selected at random from the set of all pixels
   For every stump we are selecting the best pair out of all the available pairs.
   Once we select a pair(assume the pair has values 'a' and 'b') for a stump(assume the stump belongs to class 0 here) we have to decide
    what would be our decission process - like we can have either have a<b to be 0 , a>b to be ~0 and vice versa.

2) How the program works:
    Every stump is an object of class Stump.
    Steps:
    1. Load the training data
    2. Randomly select the features
    3. Assign equal weights for all the training records
    3. For every class label create set of n ensembles
            a. for every stump in a set , first select the best pair out of the available feature pairs
                i) For every pair figure out if a<b or a>b is gonna be the classifying lever.
            b. After making a stump, calculate the weight(confidence) of the stump
            c. Now update the weights of each tranining record.
            Both the above weights are calculated using the error rate.
    4. For a new test record :
            Classify the test record with all the stumps that we have created for all the 4 types of classifiers
            Now calculate the weighted output of all the four types of classifiers
            Now which ever classifier classifies the test record with highest cumulative weight predict that particular output value

3) Problems faced:
    1. This program is easy to think but we found it a bit tricky to actually implement it.
    2. One of the challenge we faced is how to select the features out of all the available set of features


"""

class Stump(object):
    def __init__(self, pair , type,logError):
        self.pair = pair
        self.type = type # Type takes 2 values : 1 - means a<b = 0 , 2 - means a<b = ~0
        self.weight = logError

    def classify(self, row, cls):
        type = self.type
        a,b = self.pair
        pred = cls if((type == "<" and (row[a] < row[b]) ) or (type == ">" and (row[a] > row[b]))) else random.choice(list({0,90,180,270}-{cls}))
        return pred
def read_data(test_file):
    f = open(test_file,"r")
    lines = f.readlines()
    labels = []
    for i in range(len(lines)):

        lines[i] = lines[i].strip().split()
        labels.append(lines[i][0])
        lines[i] = lines[i][1:]
        for j in range(len(lines[i])):
            lines[i][j] = int(lines[i][j])
    f.close()
    return lines,labels

def findBestPair(trainData,pairsCopy,cls):
    bER = float("-inf")
    bp = pairsCopy[0] if len(pairsCopy)>0 else None
    type = "<"
    bestError = 0
    #print cls,pairsCopy
    if(len(pairsCopy)==1):
        pass
    for pair in pairsCopy:
        erRate,type,finalError = findBestErrorRate(trainData,pair,cls) #here erRate is the sum of weights of all correctly classified rows
        if(erRate > bER):
            bER = erRate
            bp = pair
            bestType = type
            bestError = finalError

    return bp,bestType,bestError

def findBestErrorRate(trainData,pair,cls):
    clsIndex = 0
    wtIndex = -1
    a,b = pair
    w1 , w2 = 0.0,0.0
    lcorr , lfalse , rcorr , rfalse = 0,0,0,0
    for row in trainData:

        if row[a] < row[b]:
            # if a<b then we are predicting 0 here
            pred1 = cls
            if(pred1 == row[clsIndex]):
                lcorr += 1
                w1 += row[wtIndex]
            else:
                lfalse += 1

            # if a<b then we are predicting ~0 here
            pred2 = row[clsIndex] if row[clsIndex] != cls else random.choice(list(allClss-{cls}))
            if pred2 == row[clsIndex]:
                rcorr += 1
                w2 += row[wtIndex]
            else:
                rfalse += 1
        elif row[a] > row[b]:
            # if a>b then we are predicting ~0 here
            pred1 = row[clsIndex] if row[clsIndex]!= cls else random.choice(list(allClss-{cls}))
            if pred1 == row[clsIndex]:
                lcorr += 1
                w1 += row[wtIndex]
            else:
                lfalse += 1

            # if a>b then we are predicting 0 here
            pred2 = cls
            if pred2 == row[clsIndex]:
                rcorr += 1
                w2 += row[wtIndex]
            else:
                rfalse += 1

    lError = float(lfalse) / (lcorr + lfalse)
    rError = float(rfalse) / (rcorr + rfalse)

    if lError < rError:
        type = "<"
        errorRate = w1
        finalError = lError
    else:
        type = ">"
        errorRate = w2
        finalError = rError
        if(lError == rError):
            #print "here both the errors are same and we took 2nd hypothesis '>' "
            pass

    return errorRate,type,finalError

def updateWeights(trainData,st,bestError,cls):
    clsIndex = 0
    wtIndex = -1
    a,b = st.pair
    type = st.type
    factor = float (bestError) / (1 - bestError)
    sumOfWeights = 0.0
    for row in trainData:
        row[wtIndex] = row[wtIndex] * factor if ((type=="<") and (( row[a] < row[b] and row[clsIndex] == cls) or ( row[a] > row[b] and row[clsIndex] != cls ))) else row[wtIndex]
        row[wtIndex] = row[wtIndex] * factor if ((type == ">") and ((row[a] > row[b] and row[clsIndex] == cls) or ( row[a] < row[b] and row[clsIndex] != cls))) else row[wtIndex]
        sumOfWeights += row[wtIndex]

    if(sumOfWeights < 1):
        remaining = 1.0 - sumOfWeights
        eachWeight = float (remaining) / len(trainData)
        for row in trainData:
            row[wtIndex] += eachWeight

    return trainData

def confusionMatrix(cm):
    #cm = [(actual,pred),(actual,pred),(actual,pred)]
    lst =   (0,90,180,270)
    matrix = [[0,0,0,0],\
              [0,0,0,0],\
              [0,0,0,0],\
              [0,0,0,0]]

    for actual,pred in cm:
        i = lst.index(actual)
        j = lst.index(pred)
        matrix[i][j] += 1


    print "Confusion Matrix is Below: (0,90,180,270)"
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            print str(matrix[i][j])+" ",
        print "\n"


allClss = {0,90,180,270}

def main(train_file , test_file, stCount):
    "python orient.py train_file.txt test_file.txt adaboost stump_count"
    "python orient.py /Users/hannavaj/Desktop/bsairamr-hannavaj-jeffravi-a5/test-data.txt /Users/hannavaj/Desktop/bsairamr-hannavaj-jeffravi-a5/test-data.txt adaboost 5"

    time1 = time.time()
    mode = "train"

    #train_file = "/Users/hannavaj/Desktop/bsairamr-hannavaj-jeffravi-a5/train-data.txt"
    trainData , labels = read_data(train_file)
    trainData = trainData[:40000]
    wt = float(1) / len(trainData)
    for i in range(len(trainData)):
        trainData[i].append(wt)


    classes = {0:[],90:[],180:[],270:[]}

    #pairs = [(3,4),(5,6),(7,8),(9,10),(11,12)]
    pairs = []
    lst = [i for i in range(1,193)]
    for i in range(stCount):
        x = random.choice(lst)
        y = random.choice(lst)
        while(x == y):
            y = random.choice(lst)
        pairs.append((x,y))

    for cls in classes:
        pairsCopy = copy.deepcopy(pairs)
        sList = []
        for i in range(stCount):
            bPair,type,bestError = findBestPair(trainData,pairsCopy,cls)
            bestError = 0.00001 if bestError == 0 else bestError
            pairsCopy.remove(bPair)
            logError = math.log((float(1 - bestError) / bestError), 2)
            st= Stump(bPair,type,logError) #@todo : Also compute the log error and send it
            trainData = updateWeights(trainData,st,bestError,cls)
            sList.append(st)
        classes[cls] = sList

    with open("model-file", 'wb') as handle:
        pickle.dump(classes, handle)

    cm = []
    corr = 0
    wrong = 0
    # test_file = "/Users/hannavaj/Desktop/bsairamr-hannavaj-jeffravi-a5/test-data.txt"
    testData, labels = read_data(test_file)
    # @todo : load the testData into testData list
    with open("model-file", 'rb') as handle:
        classes = pickle.load(handle)

    ind = 0
    f = open("adaboost_output.txt", "w")
    f.close()
    for row in testData:
        upperPredList = {}
        for cls in classes:
            sList = classes[cls]
            num = 0.0
            den = 0.0
            for st in sList:
                pred = st.classify(row, cls)
                if pred == cls:
                    num += st.weight
                den += st.weight
            upperPredList[cls] = float(num) / den

        # Below is the final prediction value for a row
        finalPred = upperPredList.keys()[upperPredList.values().index(max(upperPredList.values()))]
        cm.append((row[0], finalPred))

        f = open("adaboost_output.txt", "a")
        f.write(labels[ind] + " " + str(finalPred) + "\n")
        f.close()
        if finalPred == row[0]:
            corr += 1
        else:
            wrong += 1
        ind += 1


    confusionMatrix(cm)
    print "accuracy=", float(corr) * 100 / (corr + wrong), "%"
    print "\nNumber of stumps=",stCount
    exit()


    print "Total time taken=",time.time()-time1



