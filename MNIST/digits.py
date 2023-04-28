import numpy as np
import matplotlib as plt



def euclidian(x, y):
    distance = np.sqrt(np.sum((x - y) ** 2))
    return distance

def nearest_neighbor(x, y):
    distance = (x - y).T(x - y)
    return distance

