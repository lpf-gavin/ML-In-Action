from os import listdir
from numpy import *
from KNN import kNNclassify

class HandWriting():

    def __init__(self):
        print("init data")
        trainingFileList = listdir('trainingDigits')  # load the training set
        m = len(trainingFileList)
        self.hwLabels = []
        self.trainingMat = zeros((m, 1024))

        for i in range(m):
            fileNameStr = trainingFileList[i]
            fileStr = fileNameStr.split('.')[0]  # take off .txt
            classNumStr = int(fileStr.split('_')[0])
            self.hwLabels.append(classNumStr)
            self.trainingMat[i, :] = self.img2vector('trainingDigits/%s' % fileNameStr)

    def img2vector(self, filename):
        returnVect = zeros((1,1024))
        fr = open(filename)
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0,32*i+j] = int(lineStr[j])
        return returnVect

def test():
    handwriting = HandWriting()
    testFileList = listdir('testDigits')  # iterate through the test seterrorCount = 0.0
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = handwriting.img2vector('testDigits/%s' % fileNameStr)
        classifierResult = kNNclassify(vectorUnderTest, handwriting.trainingMat, handwriting.hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))

if __name__ == '__main__':
    test()
