import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from timeit import default_timer as timer

def main():
    indexRemoveFeature = []                                     #index of features we want to remove e.g. [0,1,2]
    C = 3                                                       #number of classes
    D = 4 - len(indexRemoveFeature)                             #number of features
    alpha  = [0.1]                                              #step factor  e.g. [0.1,0.01,0.001,0.0001]
    Ntrain = 30                                                 #number of traning data for one class
    Ntest  = 20                                                 #number of test data for one class
    NtrainAll = Ntrain*C                                        #number of traning data for all classes
    NtestAll = Ntest*C                                          #number of test data for all classes
    Niterations = 2000                                          #iterations of backpropagation
    ConfMatrixTrain = np.zeros([C,C])                           #initialize confusion matrix for train data 
    ConfMatrixTest = np.zeros([C,C])                            #initialize confusion matrix for test data 
    AllMSETrain = [[0]*Niterations for i in range(len(alpha))]  #array of all mean sqare errors
    AllMSETest =  [[0]*Niterations for i in range(len(alpha))]  #array of all mean sqare errors

 
