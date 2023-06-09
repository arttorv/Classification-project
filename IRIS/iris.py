import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from timeit import default_timer as timer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

def main():
    
    C = 3
    D = 4+1
    Ntrain = 30
    Ntest = 20
    Ntrainall = Ntrain*C
    Ntestall = Ntest*C
    Iterations = 2000
    alpha = [0.1]
    X_train, X_test, t_train, t_test = splitData('iris.data', Ntrain)
    # X_train, X_test, t_train, t_test = separateData(Ntrain)

    W = np.zeros([C,D])
    
    for m in range(Iterations):
        
        grad_mse, g_train = findMSE(X_train, t_train, W, Ntrainall, C)
        W = W - alpha*grad_mse

        g_mse, g_test = findMSE(X_test, t_test, W, Ntrainall, C)

# ------- 
    for i in range(Ntest):
        pred = np.zeros((Ntest))
        pred[i] = np.argmax(g_test[i])
    print(W)
# ---------

# FUNCTIONS
def separateData(N):
    iris = load_iris()
    size = N/50
    X_train, X_test, t_train, t_test = train_test_split(iris.data, iris.target, test_size=size)
    
    return X_train, X_test, t_train, t_test

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

def findMSE(x_loc, t_loc, W, Ntrainall, C): 
    z_loc = np.empty([Ntrainall,C])
    g_loc = np.empty([Ntrainall,C])
    grad_mse = 0

    for k in range(Ntrainall):
        z_loc[k] = np.dot(W,x_loc[k].T)
        g_loc[k] = 1/(1 + np.exp(-z_loc[k]))

        grad_mse += np.outer(((g_loc[k]-t_loc[k])*g_loc[k]*(1-g_loc[k])), x_loc[k]) 
    return grad_mse, g_loc


main()


