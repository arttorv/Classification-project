import numpy as np
import matplotlib.pyplot as plt

#The iris task

classmap = {
    'Iris-setosa': np.array([1, 0, 0]),
    'Iris-versicolor': np.array([0, 1, 0]),
    'Iris-virginica': np.array([0, 0, 1])
}

def arrange_order(x,t):
    '''
    Sets up the array such that the first 90 rows are 30*3 sets of different flowers
    '''
    array_x, array_t = [], []
    for i in range(50):
        for j in range(3):
            array_x.append(x[50*j+i])
            array_t.append(t[50*j+i])
    return np.array(array_x), np.array(array_t)

def get_x():
    f = open('iris.data')
    x = np.loadtxt(f,delimiter=',',usecols=(0,1,2,3))
    f.close()
    #[x^T 1]^T -> x
    x_new = np.ones([np.size(x,0),np.size(x,1)+1])
    x_new[:,:-1] = x
    return x_new

def get_t():
    f = open('iris.data')
    t = np.loadtxt(f,delimiter=',', usecols=(4),dtype=str)
    f.close()
    t_updated = []
    for line in np.nditer(t):   
        t_updated.append(classmap.get(str(line)))
    return t_updated

def get_data(edit_order=True):
    '''
    Returns data from the dataset in mixed or correct order
    '''
    x = get_x()
    t = get_t()
    if edit_order:
        x,t = arrange_order(x,t)
    return x,t

def MSE(A,B):
    '''
    Calculates the Mean-Squared Error between arrays A and B
    '''
    mse = (np.square(A - B)).mean(axis=1)
    return mse

def gradient_MSE(x, g, t):
    '''
    Calculates the gradient of the Mean-Squared Error
    '''
    mse_gradient = g - t
    g_gradient = g * (1-g)
    zk_gradient = x.T
    return np.dot(zk_gradient, mse_gradient*g_gradient)

def sigmoid(x):
    '''returns the sigmoid, an acceptable approximation of the heaviside function'''
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

def confusion_matrix(t_train):
    

def main():
    N = 90
    x, t = get_data()
    x_train = x[:N]
    t_train = t[:N]
    x_test = x[N:]
    t_test = t[N:]
    W, mse = train(x_train,t_train,0.01,2000)
    
    pred = predict(W,x_test)
    print(pred)
    print(len(pred))

main()


def plot_features(x, w):
    x[]
    plt.stem(x,w)
    