import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# TASK 1: IRIS

classmap = {
    'Iris-setosa': np.array([1, 0, 0]),
    'Iris-versicolor': np.array([0, 1, 0]),
    'Iris-virginica': np.array([0, 0, 1])
}

def arrange_order(x,t):
    array_x, array_t = [], []
    for i in range(50):
        for j in range(3):
            array_x.append(x[50*j+i])
            array_t.append(t[50*j+i])
    return np.array(array_x), np.array(array_t)

def get_x():
    f = open('IRIS/iris.data')
    x = np.loadtxt(f,delimiter=',',usecols=(0, 1))
    f.close()
    x_new = np.ones([np.size(x,0),np.size(x,1)+1])
    x_new[:,:-1] = x
    return x_new

def get_t():
    f = open('IRIS/iris.data')
    t = np.loadtxt(f,delimiter=',', usecols=(4),dtype=str)
    f.close()
    t_updated = []
    for line in np.nditer(t):   
        t_updated.append(classmap.get(str(line)))
    return t_updated

# Returns data from the dataset in mixed or correct order
def get_data(edit_order=True):
    x = get_x()
    t = get_t()
    if edit_order:
        x,t = arrange_order(x,t)
    return x,t

#  Calculates the Mean-Squared Error between arrays A and B
def MSE(A,B):
    mse = (np.square(A - B)).mean(axis=1)
    return mse

# Calculates the gradient of the Mean-Squared Error
def gradient_MSE(x, g, t):
    mse_gradient = g - t
    g_gradient = g * (1-g)
    zk_gradient = x.T
    return np.dot(zk_gradient, mse_gradient*g_gradient)

# returns the sigmoid, an acceptable approximation of the heaviside function
def sigmoid(x):
    return (1/(1+np.exp(-x)))
  
def train(x, t, alpha, iterations):
    '''
    x - training data,
    t - true class,
    alpha - step factor,
    W - weight matrix,
    g - output vector,
    '''
    W = np.zeros((3,x.shape[1]))
    mse_values = []
    for i in range(iterations):
        z = np.dot(x,W.T)
        g = sigmoid(z)
        W = W - alpha * gradient_MSE(x, g, t).T
        mse_values.append(MSE(g,t).mean())
    return W, mse_values 

def predict(W,x):
    g = np.dot(x, W.T)
    return np.argmax(sigmoid(g), axis=1)

def make_ref_t(t):
    t_ref = np.zeros(len(t))
    for i in range(len(t)):
        t_ref[i] = np.argmax(t[i])
    return t_ref

def make_conf_matrix(t_ref, t_pred):
    cm = confusion_matrix(t_ref, t_pred)
    return cm

def find_error(conf_matrix, N, C):
    error = 0
    for i in range(len(conf_matrix)):
        for j in range(C):
            if (j != i):
                error += conf_matrix[i][j]/N
    return error

def main():
    N = 90
    C = 3
    x, t = get_data()
    x_train = x[:N]
    t_train = t[:N]
    x_test = x[N:]
    t_test = t[N:]
    t_ref = make_ref_t(t_test)
    W, mse = train(x_train,t_train,0.04,10000)
    pred = predict(W,x_test)
    conf_matrix = make_conf_matrix(t_ref, pred)
    error = round(find_error(conf_matrix, 150-N, C)*100,2)
    
    # print(x[0:50])
    print(conf_matrix)
    print(f'Error rate: {error} %')

main()


def plot_features(x, w):
    plt.stem(x,w)
    