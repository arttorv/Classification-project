import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# TASK 1: IRIS

classes = {
    'Iris-setosa': np.array([1, 0, 0]),
    'Iris-versicolor': np.array([0, 1, 0]),
    'Iris-virginica': np.array([0, 0, 1])
}

def arrange_order(x,t):
    array_x, array_t = [], []
    for i in range(50):
        for j in range(3):
            array_x.append(x[50 * j + i])
            array_t.append(t[50 * j + i])
    return np.array(array_x), np.array(array_t)

def load_x():
    f = open('IRIS/iris.data')
    x = np.loadtxt(f,delimiter=',',usecols=(0,1,2,3))
    f.close()
    x_new = np.ones([np.size(x,0),np.size(x,1)+1])
    x_new[:,:-1] = x
    return x_new

def load_t():
    f = open('IRIS/iris.data')
    t = np.loadtxt(f,delimiter=',', usecols=(4),dtype=str)
    f.close()
    t_updated = []
    for line in np.nditer(t):   
        t_updated.append(classes.get(str(line)))
    return t_updated

# Returns data from the dataset in mixed or correct order
def get_data():
    x = load_x()
    t = load_t()
    x,t = arrange_order(x,t)
    return x,t

#  Calculates the MSE of arrays A and B
def get_MSE(A,B):
    MSE = (np.square(A - B)).mean( axis=1 )
    return MSE

# Calculates the gradient of the Mean-Squared Error
def gradient_MSE(x, g, t):
    mse_gradient = g - t
    g_gradient = g * (1-g)
    zk_gradient = x.T
    return np.dot(zk_gradient, mse_gradient*g_gradient)

# Returns the sigmoid, an acceptable approximation of the heaviside function
def sigmoid(x):
    return (1/(1+np.exp(-x)))

# Trains the LDC 
def train(x, t, alpha, iterations):
    W = np.zeros((3,x.shape[1]))
    mse_values = []
    for i in range(iterations):
        z = np.dot(x,W.T)
        g = sigmoid(z)
        W = W - alpha * gradient_MSE(x, g, t).T
        mse_values.append(get_MSE(g,t).mean())
    return W, mse_values 

# Predicts the testing sample using W
def predict(W, x):
    g = np.dot(x, W.T)
    return np.argmax(sigmoid(g), axis=1)


def make_ref_t(t):
    t_ref = np.zeros(len(t))
    for i in range(len(t)):
        t_ref[i] = np.argmax(t[i])
    return t_ref

# Makes a confusion matrix
def make_conf_matrix(t_ref, t_pred):
    cm = confusion_matrix(t_ref, t_pred)
    return cm

# Plots the confusion matrix
def plot_conf_matrix(cm, error):
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Training set \n alpha = {0.01}, Iterations = {2000} \nError rate: {round(error,2)}%')
    # Training set
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, ['Iris-setosa', 'Iris-versicolor', 'Iris-verginica'], rotation=45)
    plt.yticks(tick_marks, ['Iris-setosa', 'Iris-versicolor', 'Iris-verginica'])
    # add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Finds the error rate based on the confusion matrix
def find_error(conf_matrix, N, C):
    error = 0
    for i in range(len(conf_matrix)):
        for j in range(C):
            if (j != i):
                error += conf_matrix[i][j]/N
    return error

# Creating the histograms
def plot_feature(i):
    x = load_x()
    bins_count=30
    intervals = [(3,9), (2,5), (0.3,8),(0,3)]
    counts_0, bins0 = np.histogram(x[0:50,i], bins=bins_count, range=intervals[i])
    counts_1, bins1 = np.histogram(x[50:100,i], bins=bins_count, range=intervals[i])
    counts_2, bins2 = np.histogram(x[100:150,i], bins=bins_count, range=intervals[i])

    plt.stairs(counts_0, bins0, alpha=0.3, fill=True, label='Iris-setosa', linewidth=2, edgecolor='blue')
    plt.stairs(counts_1, bins1, alpha=0.4, fill=True, label='Iris-versicolor', linewidth=2, edgecolor='orange')
    plt.stairs(counts_2, bins2, alpha=0.3,  fill=True, label='Iris-virginica', linewidth=2, edgecolor='green')
    plt.legend()
    name = 'Width of petal leave'
    plt.xlabel(name+' [cm]')
    plt.ylabel('Samples per bin')
    plt.title(name)

# --------------------------------- #
#         RUN FUNCTIONS
# --------------------------------- #

# Runs the the code using the 30 first samples as training
def RUN_TRAIN_30FIRST():
    N = 90
    C = 3
    x, t = get_data()
    x_train = x[:N]
    t_train = t[:N]
    x_test = x[N:]
    t_test = t[N:]
    t_ref_test = make_ref_t(t_test)
    t_ref_train = make_ref_t(t_train)
    W, mse = train(x_train,t_train,0.01,2000)
    pred = predict(W,x_test)
    conf_matrix = make_conf_matrix(t_ref_test, pred)
    error = round(find_error(conf_matrix, N, C)*100,2)
    plot_conf_matrix(conf_matrix, error)
    # print(x[0:50])
    print(f'W: {W}')
    print(conf_matrix)
    print(f'Error rate: {error} %')

# Runs the the code using the 30 first samples as training
def RUN_TRAIN_30LAST():
    N = 90
    C = 3
    x, t = get_data()
    x_train = x[150-N:]
    t_train = t[150-N:]
    x_test = x[:150-N]
    t_test = t[:150-N]
    t_ref_test = make_ref_t(t_test)
    t_ref_train = make_ref_t(t_train)
    W, mse = train(x_train,t_train,0.01,2000)
    print(f'W: {W}')
    pred = predict(W,x_train)
    conf_matrix = make_conf_matrix(t_ref_train, pred)
    error = round(find_error(conf_matrix, N, C)*100,2)
    plot_conf_matrix(conf_matrix, error)
    # print(x[0:50])
    print(W)
    print(conf_matrix)
    print(f'Error rate: {error} %')

RUN_TRAIN_30FIRST()

    