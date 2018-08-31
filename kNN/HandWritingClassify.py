from os import listdir
from numpy import *

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