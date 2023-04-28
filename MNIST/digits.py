import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix

# --------------------------------- #
#            LOAD DATA 
# --------------------------------- #

def load_mnist(num_images, filename):
    with open(filename, 'rb') as f:
        f.read(16) # skip the header
        buf = f.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, 28, 28)
    return data

def load_mnist_labels(num_labels, filename):
    with open(filename, 'rb') as f:
        f.read(8) # skip the header
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return labels

def display_img(matrix):
    plt.imshow(matrix, cmap='inferno')
    plt.show()

def find_error(conf_matrix, N, C):
    error = 0
    for i in range(len(conf_matrix)):
        for j in range(C):
            if (j != i):
                error += conf_matrix[i][j]/N
    return error

# --------------------------------- #
#        EUCLIDIAN FUNCTIONS
# --------------------------------- #

def euclidian_matrix(x, y):
    distance = (x - y).T*(x - y)
    return distance

def true_distance(dist_matrix):
    sum = 0
    for i in range(len(dist_matrix)):
        for j in range(len(dist_matrix[i])):
            sum += dist_matrix[i][j]
    return sum

def nearest_neighbor(test_img, ref_images, ref_label):
    label_vec = np.zeros(len(ref_label))
    for i in range(len(ref_images)):
        dist_matrix = euclidian_matrix(test_img,ref_images[i])
        dist = true_distance(dist_matrix)
        label_vec[i]= dist
    label = ref_label[np.argmin(label_vec)]
    return label, ref_images[np.argmin(label_vec)]

def label_one(train_images, train_labels, test_image, test_label):
    pred_label, pred_image = nearest_neighbor(test_image, train_images, train_labels)
    return pred_label


# --------------------------------- #
#            CLUSTERING
# --------------------------------- #

def cluster():
    mu = 0
    for i in range():
        a = 1


# --------------------------------- #
#          RUN FUNCTIONS
# --------------------------------- #

def RUN_LABEL_ONE_IMAGE():
    N_train = 2000
    N_test = 1000
    test_number = random.randint(0,N_test)
    train_images = load_mnist(N_train, 'MNIST/train_images.bin')
    train_labels = load_mnist_labels(N_train, 'MNIST/train_labels.bin')
    test_image = load_mnist(N_test, 'MNIST/test_images.bin')[test_number]
    test_label = load_mnist_labels(N_test, 'MNIST/test_labels.bin')[test_number]
    pred_label, pred_image = nearest_neighbor(test_image, train_images, train_labels)
    print(f'tested: {test_label}')
    print(f'predicted: {pred_label}')
    display_img(test_image)
    display_img(pred_image)

    
def RUN_LABEL_MULTIPLE_IMAGES():
    Number_of_tests = 100
    N_train = 2000
    N_test = 1000
    
    train_images = load_mnist(N_train, 'MNIST/train_images.bin')
    train_labels = load_mnist_labels(N_train, 'MNIST/train_labels.bin')
    test_images = load_mnist(N_test, 'MNIST/test_images.bin')
    test_labels = load_mnist_labels(N_test, 'MNIST/test_labels.bin')
    pred_list = []
    test_list = []
    for i in range(Number_of_tests):
        test_number = random.randint(0,N_test)
        pred_list.append(label_one(train_images, train_labels, test_images[test_number], test_labels[test_number]))
        test_list.append(test_labels[test_number])
    cm = confusion_matrix(test_list, pred_list)
    error = find_error(cm, Number_of_tests, 10)
    print(cm)
    print(error)

def RUN_EUCLIDIAN():
    # Example usage: Load 1000 images
    N = 20
    images = load_mnist(N)
    labels = load_mnist_labels(N)
    image1 = images[2]
    image2 = images[10]
    dist_matrix = euclidian_matrix(image1,image2)
    dist = true_distance(dist_matrix)
    print(dist)
    print(labels)
    display_img(image1)

RUN_LABEL_MULTIPLE_IMAGES()