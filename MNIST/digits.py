import numpy as np
import matplotlib.pyplot as plt
import random


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

# --------------------------------- #
#        EUCLIDIAN FUNCTIONS
# --------------------------------- #

def euclidian(x, y):
    distance = np.sqrt(np.sum((x - y) ** 2))
    return distance

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
    display_img(ref_images[np.argmin(label_vec)])
    return label


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
def LABEL_ONE_IMAGE():
    N = 2000
    test_number = random.randint(0,N)
    train_images = load_mnist(N, 'MNIST/train_images.bin')
    train_labels = load_mnist_labels(N, 'MNIST/train_labels.bin')
    test_image = load_mnist(N, 'MNIST/test_images.bin')[test_number]
    test_label = load_mnist_labels(N, 'MNIST/test_labels.bin')[test_number]
    pred_label = nearest_neighbor(test_image, train_images, train_labels)
    
    display_img(test_image)
    print(test_label)
    print(pred_label)



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

LABEL_ONE_IMAGE()