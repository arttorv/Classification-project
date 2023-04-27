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
    D = 4
    N = 30
    Iterations = 1000
    alpha = 0.1
    X_train, X_test, t_train, t_test = separateData(N)
    W = np.zeros([C,D])
    
    for m in range(Iterations):
        
        g_mse, g_train = findMSE(X_train, t_train, W, N, C)
        W = W - alpha*g_mse

        g_mse, g_test = findMSE(X_test, t_test, W, N, C)

# ------- 
    for i in range(50-N):
        pred = np.zeros(50-N)
        pred[i] = np.argmax(g_test[i])
    print(g_test)
# ---------

# FUNCTIONS
def separateData(N):
    iris = load_iris()
    size = N/50
    X_train, X_test, t_train, t_test = train_test_split(iris.data, iris.target, test_size=size)
    return X_train, X_test, t_train, t_test

def findMSE(x_loc, t_loc, W, N, C): 
    z_loc = np.zeros([N,C])
    g_loc = np.zeros([N,C])
    grad_mse = 0

    for k in range(N):
        z_loc[k] = np.dot(W,x_loc[k].T)
        g_loc[k] = 1/(1 + np.exp(-z_loc[k]))

        grad_mse += np.outer(((g_loc[k]-t_loc[k])*g_loc[k]*(1-g_loc[k])), x_loc[k]) 
    return grad_mse, g_loc

main()


