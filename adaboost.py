import math
import random
import pickle
import copy
import time
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
    for i in range(len(lines)):
        lines[i] = lines[i].strip().split()[1:]
        for j in range(len(lines[i])):
            lines[i][j] = int(lines[i][j])
    f.close()
    return lines

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

allClss = {0,90,180,270}

if __name__ == "__main__":
    time1 = time.time()
    stCount = 25
    mode = "train"
    if mode == "test":
        corr = 0
        wrong = 0
        test_file = "/Users/hannavaj/Desktop/bsairamr-hannavaj-jeffravi-a5/test-data.txt"
        testData = read_data(test_file)
        #@todo : load the testData into testData list
        with open("adaboost", 'rb') as handle:
            classes = pickle.load(handle)

        for row in testData:
            upperPredList = {}
            for cls in classes:
                sList = classes[cls]
                num = 0.0
                den = 0.0
                for st in sList:
                    pred = st.classify(row,cls)
                    if pred == cls:
                        num += st.weight
                    den += st.weight
                upperPredList[cls] = float(num)/den
            #Below is the final prediction value for a row
            finalPred = upperPredList.keys()[upperPredList.values().index(max(upperPredList.values()))]
            if finalPred == row[0]:
                corr += 1
            else:
                wrong += 1
        print corr,wrong
        print "accuracy=",float(corr)/(corr + wrong)
        exit()

    train_file = "/Users/hannavaj/Desktop/bsairamr-hannavaj-jeffravi-a5/train-data.txt"
    trainData = read_data(train_file)
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

    with open("adaboost", 'wb') as handle:
        pickle.dump(classes, handle)


print "Total time taken=",time.time()-time1



    #pairs = generateRandomPairs()

