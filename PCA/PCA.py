'''
"Machine Learning In Action" Chapter13 P246
'''

from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(filename, delim='\t'):
    fr = open(filename)
    strArr = [line.strip().split(delim) for line in fr.readlines()]
    # dataArr = [map(float, line) for line in strArr]
    matRes = [[float(s) for s in line] for line in strArr]
    return mat(matRes)

def PCA(dataMat, topNfeat=9999999):
    '''
    for a m * n matrix
    no value for axis,return mean value of m * n variables
    axis = 0：return a 1 * n matrix, represents average value for each column
    axis =1: return a m * 1 matrix, represents average value for each row
    '''

    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    '''
    If rowvar is True (default), each row represents a variable and is displayed in a column(that is, one sample per column).
    Otherwise, relationships are transposed: each column represents variables, and rows contain observations.
    '''

    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValIdx = argsort(eigVals)
    eigValIdx = eigValIdx[:-(topNfeat + 1):-1]
    redEigVects = eigVects[:,eigValIdx]
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat #reconMat for test



if __name__ == '__main__':
    dataMat = loadDataSet('testSet.txt')
    lowDMat, reconMat = PCA(dataMat, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('PCA')
    ax.scatter(dataMat[:,0].tolist(), dataMat[:,1].tolist(), marker='.', s=90)
    ax.scatter(reconMat[:,0].tolist(), reconMat[:,1].tolist(), marker='o', s=50, c='red')
    plt.show()