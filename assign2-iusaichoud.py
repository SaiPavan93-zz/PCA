from __future__ import division
import numpy as np
from numpy import linalg as LA
import math
import sys

def zScore(matrix):
    matrix_transpose = matrix.transpose()
    meanmat=matrix_transpose.mean(1)
    stdmat=matrix_transpose.std(1)
    meanexp=np.outer(meanmat,np.ones(19020))
    meaned_matrix=matrix_transpose-meanexp
    zscore_mat=np.divide(meaned_matrix,stdmat)
    return(zscore_mat.transpose())

def covariance(matrix):
    meanmat=matrix.mean(0)
    meanexp=np.outer(meanmat.transpose(),np.ones(19020))
    mean=meanexp.transpose()
    centereddataMatrix=matrix-mean
    n=centereddataMatrix.shape[0]
    sum=0
    for i in range(centereddataMatrix.shape[0]):
        sum=sum+(centereddataMatrix[i].transpose().dot(centereddataMatrix[i]))
    cov=(sum/n)
    covar=np.cov(matrix,rowvar=False,bias=True)
    return cov,covar

def powerIterator(matrix,x0,converge,eigen):
    evalue = max(x0,key=abs)
    x0=x0/evalue
    evector=x0
    converge.append(evalue)
    eigen.append(evector)
    while(True):
        x1=np.matmul(matrix,x0)
        evalue=max(x1,key=abs)
        x1=x1/evalue

        a = x1-x0
        sum1 = 0
        for each in a:
            sum1 = sum1 + math.pow(each, 2)
        sqr = math.sqrt(sum1)

        if (sqr < 0.000001):
            break
        x0=x1
    evector=(x1/np.linalg.norm(x1))
    values,vectors=LA.eig(matrix)
    return evalue,evector,values,vectors.transpose()

def projection(newData,covmat):
    c=LA.eig(covmat)
    vectors=c[1].transpose()
    a=c[0]
    d=c[0]
    d=d.tolist()
    a=a.tolist()
    first=max(a)
    indexfirst=a.index(first)
    a.remove(first)
    second=max(a)
    for each in d:
        if(each==second):
            indexsecond = d.index(second)
    U1=vectors[indexfirst]
    U2=vectors[indexsecond]
    U=np.matmul(U1.transpose(),U1)
    UK=np.matmul(U2.transpose(),U2)
    UU=U+UK
    data = np.matmul(newData, UU.transpose())
    mean = np.mean(data, axis=0)
    mn = np.outer(mean, np.ones(newData.shape[0]))
    centered = data - mn.transpose()
    v = np.matmul(centered.transpose(), centered)
    variance = v / newData.shape[0]
    return (sum(np.diagonal(variance)))

def eigenDecomposition(covmat):
    values,vectors=LA.eig(covmat)
    mat=vectors.transpose()
    digmat=np.diag(values)
    #print(vectors.dot(digmat).dot(mat))
    #print(covmat)
    return vectors.transpose(),digmat,vectors

def pca(matrix):
    mean_mat=np.mean(matrix,axis=0)
    mean=np.outer(mean_mat,np.ones(matrix.shape[0]))
    centeredmat=matrix-mean.transpose()
    cov=(np.matmul(centeredmat.transpose(),centeredmat))/centeredmat.shape[0]
    evalues,evectors=LA.eigh(cov)
    evectors=evectors.transpose()
    totalvariance=sum(evalues)
    alpha=0.95
    a=[]
    sum1=0
    ev=[]
    evect=[]
    ev=evalues
    evalues=evalues.tolist()
    ev=sorted(ev,reverse=True)
    for each in ev:
        sum1=sum1+each
        a.append(sum1/totalvariance)
    count=1
    for each in a:
        if(each<=0.95):
            count=count+1
    for each in ev[0:count]:
        for j in evalues:
            if(each==j):
                evect.append(evectors[evalues.index(j)])
    evect=np.concatenate(evect)
    U=evect
    cp=[]
    cordinatepoints=(matrix.dot(U.transpose()))
    cp=np.concatenate(cordinatepoints)
    cvar=np.cov(cp,rowvar=False,bias=True)
    #print(sum(ev[0:count]))
    tr=np.diag(cvar)
    #print(sum(tr))
    return cp[0:10],sum(ev[0:count]),sum(tr)

def main():
    file=open(sys.argv[1],"r")
    f=file.read()
    data=f.split("\n")
    #print(data)
    dataset=[]
    a=[]
    diff = []
    converge = []
    eigen = []
    for each in data:
        a.append(each.split(","))
    for each in a:
        k = []
        #print(each[0:len(each)-1])
        for i in each[0:len(each)-1]:
            k.append(float(i))
        dataset.append(k)

    sys.stdout = open("assign2-iusaichoud.txt", "w")
    c=np.matrix(dataset)
    newDataset=zScore(c)
    print("a","\n\n","Z-SCORE NORMALIZED DATA IS",newDataset)
    covmat,covmatwithfunction=covariance(newDataset)
    print("\n\n\n","b","\n\n","Calculated Covariance is :",covmat,"\n","Covariance with Function is :",covmatwithfunction)
    values,vector,functionValues,functionVectors=powerIterator(covmat,covmat[0].transpose(),converge,eigen)
    print("\n\n\n","c","\n\n","Calculated Dominant Eigen value is :",values,"\n\n","Calculated Eigen vector is :",vector,"\n\n","Eigen Values Calculated with function :",
          functionValues,"\n\n","Eigen vectors Calculated with function is :",functionVectors)
    varianceProjectedSpace=projection(newDataset,covmat)
    print("\n\n\n","d","\n\n","The variance of the datapoints in the projected subspace is :",varianceProjectedSpace)
    vectors,diagonalMatrix,vectorstranspose=eigenDecomposition(covmat)
    print("\n\n\n","e","\n\n","U*A*U(T) :",vectors,diagonalMatrix,vectorstranspose)
    coordinatepoints,varianceofData,calculatedVariance=pca(newDataset)
    print("\n\n\n","f","\n\n","co-ordinate of the first 10 data points",coordinatepoints,"\n\n\n","g","\n\n","Variance from data is :",varianceofData,"\n","Calculated Variance of the projected data points :",calculatedVariance)

if __name__=="__main__":
    main()
