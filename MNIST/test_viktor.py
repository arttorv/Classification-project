import numpy as np
import matplotlib as plt
from sklearn.cluster import KMeans

def load_mnist(num_images):
    with open('MNIST/train_images.bin', 'rb') as f:
        f.read(16) # skip the header
        buf = f.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, 28, 28)
    return data

def load_mnist_labels(num_labels):
    with open('MNIST/train_labels.bin', 'rb') as f:
        f.read(8) # skip the header
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return labels

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

# ''''''''''''''''''''''''''''''''''''''''''''''''''
# TEST CLUSTERING

# ''''''''''''''''''''''''''''''''''''''''''''''''''


def cluster(x, t): #feature matrix x, target/label matrix y_t
    kmeans = KMeans(n_clusters=10, random_state=0)
    
    kmeans.fit(x)
    
    cluster_center = kmeans.cluster_centers_
    
    inertia = kmeans.inertia_

    
    
    
    # Fra GPT
    # Visualize the cluster centers as images
    fig, ax = plt.subplots(2, 5, figsize=(8, 3))
    centers = centers.reshape(10, 8, 8)
    for i, axi in enumerate(ax.flat):
        axi.imshow(centers[i], cmap=plt.cm.binary)
        axi.set(xticks=[], yticks=[])
    
    mu = 0
    for i in range():
        a = 1




# ''''''''''''''''''''''''''''''''''''''''''''''''''
#------------------------------------------------

# ''''''''''''''''''''''''''''''''''''''''''''''''''





def RUN_EUCLIDIAN():
    # Example usage: Load 1000 images
    N = 12
    images = load_mnist(N)
    labels = load_mnist_labels(N)
    image1 = images[2]
    image2 = images[10]
    dist_matrix = euclidian_matrix(image1,image2)
    dist = true_distance(dist_matrix)
    print(dist)
    print(labels)
