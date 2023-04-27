import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from timeit import default_timer as timer

def main():
    indexRemoveFeature = []                                     #index of features we want to remove e.g. [0,1,2]
    C = 3                                                       #number of classes
    D = 4 - len(indexRemoveFeature)                             #number of features
    alpha  = [0.2]                                              #step factor  e.g. [0.1,0.01,0.001,0.0001]
    Ntrain = 40                                                 #number of traning data for one class
    Ntest  = 10                                                 #number of test data for one class
    NtrainAll = Ntrain*C                                        #number of traning data for all classes
    NtestAll = Ntest*C                                          #number of test data for all classes
    Niterations = 1000                                          #iterations of backpropagation
    ConfMatrixTrain = np.zeros([C,C])                           #initialize confusion matrix for train data 
    ConfMatrixTest = np.zeros([C,C])                            #initialize confusion matrix for test data 
    AllMSETrain = [[0]*Niterations for i in range(len(alpha))]  #array of all mean sqare errors
    AllMSETest =  [[0]*Niterations for i in range(len(alpha))]  #array of all mean sqare errors

    #-----------THE DIFFERENT CASES THAT ARE TESTED-----------#
    #xTrain, xTest, tTrain, tTest = splitData('iris.data', Ntrain, indexRemoveFeature) #trained using a first 30 samples
    xTest, xTrain, tTest, tTrain = splitData('iris.data', Ntest, indexRemoveFeature) #trained using a last 30 samples
    
    #-----------THE CLASSIFIER IS TRAINED AND THE MSE IS FOUND-----------#
    for j in range(len(alpha)):
        W = np.zeros((C,D+1))   
        for i in range(Niterations): 

            zTrain, gTrain, MSETrain, GradMSETrain = findMSE(xTrain,tTrain, NtrainAll, C, W)

            W = W - alpha[j]*GradMSETrain
            
            AllMSETrain[j][i] = MSETrain

            zTest, gTest, MSETest, GradMSETest = findMSE(xTest, tTest, NtestAll, C, W)

            AllMSETest[j][i] = MSETest
        plt.plot(AllMSETest[j], label=f"Testing data \u03B1={alpha[j]}") #plots the MSE for each alpha
    plt.xlabel('iterations')
    plt.ylabel('MSE')
    plt.title('MSE for differenet \u03B1')
    plt.legend()

    pred = np.zeros((NtestAll))
    for i in range(NtestAll):
        pred[i] = np.argmax(gTest[i])
    print(pred)






def splitData(fileName, N, col = None):
    df = pd.read_csv(fileName, header = None)
    df[4] = 1 

    if (col != None):
        for i in col:
            df.drop(i, inplace=True, axis=1)
    
    df_S    = df.iloc[   :50 ]
    df_Vc   = df.iloc[50 :100]
    df_Vg   = df.iloc[100:150]


    df_S_train  =  df_S.iloc[:N]
    df_S_test   =  df_S.iloc[N:]
    df_Vc_train = df_Vc.iloc[:N]
    df_Vc_test  = df_Vc.iloc[N:]
    df_Vg_train = df_Vg.iloc[:N]
    df_Vg_test  = df_Vg.iloc[N:]

    firstX      = np.concatenate((df_S_train,df_Vc_train,df_Vg_train), axis=0) #training data 
    secondX     = np.concatenate((df_S_test,df_Vc_test,df_Vg_test), axis=0) #testing data

    firstTS     = np.tile(np.array([[1,0,0]]), (N, 1))
    firstTVc    = np.tile(np.array([[0,1,0]]), (N, 1))
    firstTVg    = np.tile(np.array([[0,0,1]]), (N, 1))
    firstT      = np.concatenate((firstTS,firstTVc,firstTVg),axis=0)               

    secondTS    = np.tile(np.array([[1,0,0]]), (50-N, 1))
    secondTVc   = np.tile(np.array([[0,1,0]]), (50-N, 1))
    secondTVg   = np.tile(np.array([[0,0,1]]), (50-N, 1))
    secondT     = np.concatenate((secondTS,secondTVc,secondTVg),axis=0)

    return firstX, secondX, firstT, secondT

def findMSE(x_local, t_local, N, C, W):
    z_local = np.empty([N,C]) 
    g_local = np.empty([N,C]) 
    MSE_local = 0
    GradMSE_local = 0 
    for k in range(N):
        z_local[k]=np.dot(W,x_local[k].T)  

        g_local[k]=1/(1+np.exp(-z_local[k]))

        MSE_local += 0.5*np.dot((g_local[k]-t_local[k]).T,(g_local[k]-t_local[k]))
        GradMSE_local += np.outer(((g_local[k]-t_local[k])*g_local[k]*(1-g_local[k])), x_local[k]) 
    return z_local, g_local, MSE_local, GradMSE_local

main()