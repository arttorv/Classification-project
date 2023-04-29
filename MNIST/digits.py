import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from statistics import mode, StatisticsError
from collections import Counter
from PIL import Image

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

def most_common(lst):
    c = Counter(lst)
    most_common = c.most_common()
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        for item in lst:
            if item == most_common[0][0] or item == most_common[1][0]:
                return item
    else:
        return most_common[0][0]

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

# RETURNS THE PREDICTED LABEL AND THE PICTURE THAT LOOKS MOST LIKE THE TEST
def nearest_neighbor(test_img, ref_images, ref_label):
    label_vec = np.zeros(len(ref_label))
    for i in range(len(ref_images)):
        dist_matrix = euclidian_matrix(test_img,ref_images[i])
        dist = true_distance(dist_matrix)
        label_vec[i]= dist
    label = ref_label[np.argmin(label_vec)]
    return label, ref_images[np.argmin(label_vec)], label_vec


def k_nearest_neighbor(test_img, ref_images, ref_label, k):
    k_nearest = []
    label, ref_images, dist_vec = nearest_neighbor(test_img, ref_images, ref_label)
    for i in range(k):
        min_index = np.argmin(dist_vec)
        k_nearest.append(ref_label[min_index])
        np.delete(dist_vec,min_index)
    pred_label = most_common(k_nearest)
    return pred_label
        
         
# DENNE FUNSKJONEN ER LITT DUST MEN FUNKER
def label_one(train_images, train_labels, test_image, test_label):
    pred_label, pred_image = nearest_neighbor(test_image, train_images, train_labels)
    return pred_label


# --------------------------------- #
#            CLUSTERING
# --------------------------------- #


def data_for_clustering(train_images, train_labels):
    num_samples=int(len(train_labels)*0.9)
    sorted_images_in_rows = np.zeros((10,len(train_labels),784))  ##bytte med numsamples
    class_count=np.zeros(10)
    for i in range(len(train_labels)):
        temp_image = np.zeros(784)
        for j in range(28):
            for k in range(28):
                temp_image[j*28+k]=train_images[i][j][k]
        
        label=int(train_labels[i])
        print(label)

        count=int(class_count[label])
        
        sorted_images_in_rows[label][count]=temp_image
        
        class_count[label]=class_count[label]+1
        
    return sorted_images_in_rows, num_samples
        

def cluster(class_images): #feature matrix x, target/label matrix y_t
    kmeans = KMeans(n_clusters=64)#, random_state=0)
    
    # clustering 
    
    kmeans.fit(class_images)
    
    cluster_centers = kmeans.cluster_centers_
    
    #inertia = kmeans.inertia_
    return cluster_centers


def row_to_image(row):
    local_image=np.zeros((28,28))
    for i in range(28):
        for j in range(28):
            local_image[i][j]=row[i*28+j]
    return local_image
    
       
def get_new_clustered_images(sorted_images_in_rows, N_cluster):
    train_images_clustered=np.zeros((N_cluster*10,28,28))
    train_labels_clustered=[]
    for i in range(10):
        for k in range(len(sorted_images_in_rows[i])):
            print('her', sorted_images_in_rows[i])
            clustered_class = cluster(sorted_images_in_rows[i])
            for j in range(N_cluster):
                image_local=row_to_image(clustered_class[j])
                train_images_clustered[i*N_cluster+j] = image_local
                # train_images_clustered[i]=cluster(sorted_images_in_rows[i])
    return train_images_clustered
        

def get_new_clustered_labels(sorted_images_in_rows):
    train_labels_clustered_list = []
    for i in range(10):
        num_images_for_class = len(sorted_images_in_rows[i])
        train_labels_clustered_list.append(np.full(num_images_for_class,i))
        np.array(train_labels_clustered_list)
    return train_labels_clustered_list
    
    
# def plot_clusters(clusters):
#     fig, ax = plt.subplots(2, 5, figsize=(8, 3))
#     centers = centers.reshape(10, 8, 8)
#     for i, axi in enumerate(ax.flat):
#         axi.imshow(centers[i], cmap=plt.cm.binary)
#         axi.set(xticks=[], yticks=[])

# --------------------------------- #
#          RUN FUNCTIONS
# --------------------------------- #

def import_custom_BMP(filename):
    bmp_image = Image.open(filename)
    # Convert the PIL image to a NumPy array
    np_array = np.array(bmp_image)
    # Print the shape of the NumPy array
    print(np_array.shape)
    display_img(np.array)
    

# NN
def RUN_LABEL_ONE_NN():
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

# KNN
def RUN_LABEL_ONE_KNN():
    k = 7
    N_train = 2000
    N_test = 100
    test_number = random.randint(0,N_test)
    train_images = load_mnist(N_train, 'MNIST/train_images.bin')
    train_labels = load_mnist_labels(N_train, 'MNIST/train_labels.bin')
    test_image = load_mnist(N_test, 'MNIST/test_images.bin')[test_number]
    test_label = load_mnist_labels(N_test, 'MNIST/test_labels.bin')[test_number]
    pred_label = k_nearest_neighbor(test_image,train_images,train_labels,k)
    print(f'tested: {test_label}')
    print(f'predicted: {pred_label}')

# PRINTING CONFUSION MATRIX OG ERROR RATE USING NN   
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

# SIMPLE FIRST TEST OF THE FUNCTIONS (REMOVE THIS)
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

def RUN_CLUSTER_TEST():
    N_train = 200
    N_test = 1000
    # test_number = random.randint(0,N_test)
    train_images = load_mnist(N_train, 'train_images.bin')
    train_labels = load_mnist_labels(N_train, 'train_labels.bin')
    print(train_labels)
    # test_image = load_mnist(N_test, 'MNIST/test_images.bin')[test_number]
    # test_label = load_mnist_labels(N_test, 'MNIST/test_labels.bin')[test_number]
    # pred_label, pred_image = nearest_neighbor(test_image, train_images, train_labels)
    # print(f'tested: {test_label}')
    # print(f'predicted: {pred_label}')
    # display_img(clustered_image)
    clust_data = data_for_clustering(train_images, train_labels)
    get_new_clustered_images(clust_data, 64)
    get_new_clustered_labels(clust_data)
    print(clust_data)
    display_img(clust_data[0])
    display_img(clust_data[1])
    display_img(clust_data[2])
    display_img(clust_data[64])
    display_img(clust_data[65])
    
# import_custom_BMP('Test9tall.bmp')

RUN_CLUSTER_TEST()
