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
    
    alpha = 1
    X_train, X_test, t_train, t_test = separateData()




    print(t_train)


# FUNCTIONS
def separateData():
    iris = load_iris()

    X_train, X_test, t_train, t_test = train_test_split(iris.data, iris.target, test_size=0.5)
    return X_train, X_test, t_train, t_test

def findMSE(): 

    mse = []
    grad_mse = []


    return mse, grad_mse

main()


