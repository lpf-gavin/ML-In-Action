from HandWritingClassify import *
from KNN import kNNclassify

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
