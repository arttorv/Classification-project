import numpy as np
import matplotlib as plt

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

# Example usage: Load 1000 images
images = load_mnist(9)
labels = load_mnist_labels(9)
image1 = images[3]
image2 = images[4]
dist_matrix = euclidian_matrix(image1,image2)
true_distance =
print(dist_matrix)
print(labels)