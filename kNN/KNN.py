'''

Machine Learning In Action Chapter2
Using kNN to solve handwriting classify problem. 
'''

from numpy import *
# assume n is dimension of data, the shape of inX is 1 * n, m * n for dataSet, m for labels
def kNNclassify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    # axis=1: solve every row
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    resLabel = None
    maxCnt = 0
    for it in classCount.keys():
        if classCount[it] > maxCnt:
            maxCnt = classCount[it]
            resLabel = it
    return resLabel