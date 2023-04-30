import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from statistics import mode, StatisticsError
from collections import Counter
from PIL import Image
from scipy.spatial import distance
import time

# TASK 2: DIGITS

# --------------------------------- #
#       LOAD DATA FUNCTIONS
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

# FOR IMPORTING SELF-WRITTEN NUMBERS
def import_custom_BMP(filename):
    bmp_image = Image.open(filename)
    # Convert the PIL image to a NumPy array
    np_array = np.array(bmp_image)
    # Print the shape of the NumPy array
    print(np_array.shape)
    display_img(np.array)


# --------------------------------- #
#       DISPLAY FUNCTIONS
# --------------------------------- #

def display_img(matrix):
    plt.imshow(matrix, cmap='inferno')
    plt.show()

def plot_conf_matrix(cm, error):
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(f'KNN with clustering \nError rate: {round(error*100,2)}%')
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, ['Digit 0', 'Digit 1', 'Digit 2', 'Digit 3', 'Digit 4','Digit 5','Digit 6','Digit 7','Digit 8','Digit 9'], rotation=45)
    plt.yticks(tick_marks, ['Digit 0', 'Digit 1', 'Digit 2', 'Digit 3', 'Digit 4','Digit 5','Digit 6','Digit 7','Digit 8','Digit 9'])

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

    
def display_clustered_digit(digit_to_display):
    # N_train = 1000
    # train_images = load_mnist(N_train, 'MNIST/train_images.bin')
    clustered_images = np.load('MNIST/clustered_templates.npy')
    fig, axs = plt.subplots(8, 8, sharex=True, sharey=True)
    for i in range(8): 
         for j in range(8): 
            axs[i,j].imshow(clustered_images[64*digit_to_display + i*8+j], cmap='inferno')
            axs[i,j].axis('off')
    fig.set_size_inches(4,4)
    plt.show()

# --------------------------------- #
#        EUCLIDIAN FUNCTIONS
# --------------------------------- #


def euclidian_matrix(x, y): #matrix definition of euclidean distance
    distance = (x - y).T*(x - y)
    return distance

def true_distance(dist_matrix): #finds euclidean distance to NN
    sum = 0
    for i in range(len(dist_matrix)):
        for j in range(len(dist_matrix[i])):
            sum += dist_matrix[i][j]**2
    np.sqrt(sum)
    return sum

def most_common(lst):
    c = Counter(lst)
    most_common = c.most_common()
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        for item in lst:
            if item == most_common[0][0] or item == most_common[1][0]:
                return item
    else:
        return most_common[0][0]
    
def find_error(conf_matrix, N, C):
    error = 0
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[i])):
            if (j != i):
                error += conf_matrix[i][j]/N
    return error

# INITIAL FUNCTION FOR NN USING DEFINITION OF EUCLIDEAN DIST
def nearest_neighbor(test_img, ref_images, ref_label):
    label_vec = np.zeros(len(ref_label))
    # print('labelveclen',len(label_vec))
    for i in range(len(ref_images)):
        dist_matrix = euclidian_matrix(test_img,ref_images[i]) # finding distance for features of the test_img and training data
        dist = true_distance(dist_matrix) # finding euclidean distance between features of the test_img and training data
        label_vec[i]= dist
    label = ref_label[np.argmin(label_vec)]
    return label, ref_images[np.argmin(label_vec)], label_vec

# RETURNS THE PREDICTED LABEL AND ITS NN
def nearest_neighbor_scipy(test_img, ref_images, ref_label):
    n_train_images = len(ref_images)
    label_vec = np.zeros(len(ref_label))
    test_img = test_img.reshape(784)
    ref_images = np.array(ref_images).reshape(n_train_images,784)
    for i in range(n_train_images):
        dist = distance.euclidean(test_img,ref_images[i]) #returns euclidean distance between features of the test_img and training data
        label_vec[i]= dist
    label = ref_label[np.argmin(label_vec)] #finds label of NN
    return label, ref_images[np.argmin(label_vec)], label_vec

def k_nearest_neighbor(test_img, train_images, train_label, k):
    k_nearest = []
    # label, ref_image, dist_vec = nearest_neighbor(test_img, train_images, train_label)
    label, ref_image, dist_vec = nearest_neighbor_scipy(test_img, train_images, train_label)
    for i in range(k):
        min_index = np.argmin(dist_vec) #index of NN
        k_nearest.append(train_label[min_index]) #adds label of NN to list of KNN
        dist_vec[min_index] = 1000000000  #makes NN not NN anymore
    print(k_nearest)
    pred_label = most_common(k_nearest)
    return pred_label


def label_one(train_images, train_labels, test_image, test_label):
    pred_label, pred_image, dummy_vec = nearest_neighbor(test_image, train_images, train_labels)
    return pred_label


# --------------------------------- #
#       CLUSTERING FUNCTIONS
# --------------------------------- #


def data_for_clustering(train_images, train_labels):
    sorted_images_in_rows = np.zeros((10,len(train_labels),784))  ##bytte med numsamples
    class_count=np.zeros(10)
    
    for i in range(len(train_labels)):
        temp_image_row = np.zeros(784)
        for j in range(28):
            for k in range(28):
                temp_image_row[j*28+k]=train_images[i][j][k]
        
        label=int(train_labels[i])
        count=int(class_count[label])
        sorted_images_in_rows[label][count]=temp_image_row
        class_count[label]=class_count[label]+1
    
    return sorted_images_in_rows

def cluster_class(class_images):
    kmeans = KMeans(n_clusters=64)#, random_state=0)
    kmeans.fit(class_images)
    clustered_centers = kmeans.cluster_centers_
    return clustered_centers
    
def get_new_clustered_images(sorted_images_in_rows, N_cluster):
    train_images_clustered=np.zeros((N_cluster*10,28,28))
    train_labels_clustered=[]
    for i in range(10):
        clustered_class = cluster_class(sorted_images_in_rows[i])
        for j in range(N_cluster):
            image_local=row_to_image(clustered_class[j])
            train_images_clustered[i*N_cluster+j] = image_local
            # train_images_clustered[i]=cluster_class(sorted_images_in_rows[i])
    return train_images_clustered

def row_to_image(row):
    local_image=np.zeros((28,28))
    for i in range(28):
        for j in range(28):
            local_image[i][j]=int(row[i*28+j])
    return local_image

def get_new_clustered_labels(N_cluster,n_digits):
    train_labels_clustered_list = []
    for i in range(n_digits):
        for j in range(N_cluster):
            train_labels_clustered_list.append(i)
    train_labels_clustered_list=np.array(train_labels_clustered_list)
    print(train_labels_clustered_list)
    np.save('MNIST/clustered_labels.npy', train_labels_clustered_list)
    return train_labels_clustered_list

# --------------------------------- #
#          RUN FUNCTIONS
# --------------------------------- #

# SIMPLE FIRST TEST COMPARING TWO IMAGE
def RUN_EUCLIDIAN():
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

# NN
def RUN_LABEL_ONE_NN():
    N_train = 2000
    N_test = 1000
    test_number = random.randint(0,N_test)
    train_images = load_mnist(N_train, 'MNIST/train_images.bin')
    train_labels = load_mnist_labels(N_train, 'MNIST/train_labels.bin')
    train_images = np.load('MNIST/clustered_templates.npy')
    train_labels = np.load('MNIST/clustered_labels.npy')
    test_image = load_mnist(N_test, 'MNIST/test_images.bin')[test_number]
    test_label = load_mnist_labels(N_test, 'MNIST/test_labels.bin')[test_number]
    pred_label, pred_image, dummy_vec = nearest_neighbor(test_image, train_images, train_labels)
    print(f'tested: {test_label}')
    print(f'predicted: {pred_label}')
    display_img(test_image)
    display_img(pred_image)

# KNN
def RUN_LABEL_ONE_KNN():
    k = 7
    N_train = 10000
    N_test = 100
    test_number = random.randint(0,N_test)
    # train_images = load_mnist(N_train, 'MNIST/train_images.bin')
    # train_labels = load_mnist_labels(N_train, 'MNIST/train_labels.bin')
    train_images = np.load('MNIST/clustered_templates.npy')
    train_labels = np.load('MNIST/clustered_labels.npy')
    print(train_labels)
    test_image = load_mnist(N_test, 'MNIST/test_images.bin')[test_number]
    test_label = load_mnist_labels(N_test, 'MNIST/test_labels.bin')[test_number]
    pred_label = k_nearest_neighbor(test_image,train_images,train_labels,k)
    print(f'tested: {test_label}')
    print(f'predicted: {pred_label}')

# PRINTING CONFUSION MATRIX OG ERROR RATE USING NN WITH ALL TESTING DATA
def RUN_LABEL_MULTIPLE_IMAGES_NN(): 
    Number_of_tests = 1000
    N_train = 2000
    N_test = 1000
    # train_images = np.load('MNIST/clustered_templates.npy')
    # train_images = int_cluster(train_images)
    # train_labels = np.load('MNIST/clustered_labels.npy')
    train_images = load_mnist(N_train, 'MNIST/train_images.bin')
    train_labels = load_mnist_labels(N_train, 'MNIST/train_labels.bin')
    test_images = load_mnist(N_test, 'MNIST/test_images.bin')
    test_labels = load_mnist_labels(N_test, 'MNIST/test_labels.bin')
    pred_list = []
    test_list = []
    start_time_KNN=time.time()
    for i in range(Number_of_tests-1):
        test_number = random.randint(0,N_test-1)
        pred_label = label_one(train_images, train_labels, test_images[i], test_labels[i])
        pred_list.append(pred_label)
        test_list.append(test_labels[i])
        print(f'tested: {test_labels[i]}')
        print(f'predicted: {pred_label}')
    cm = confusion_matrix(test_list, pred_list)
    error = find_error(cm, Number_of_tests, 10)
    print(cm)
    print(error)
    finish_time_KNN=time.time()
    print('Time nn: ', finish_time_KNN-start_time_KNN)
    plot_conf_matrix(cm, error)
    


# PRINTING CONFUSION MATRIX OG ERROR RATE USING KNN WITH ALL TESTING DATA
def RUN_LABEL_MULTIPLE_IMAGES_KNN(): # NOW: not clustered
    Number_of_tests = 1000
    N_train = 20000
    N_test_images = 1000
    k = 7
    # train_images = int_cluster(np.load('MNIST/clustered_templates.npy')  )          # clustered
    # train_labels = np.load('MNIST/clustered_labels.npy')              # clustered
    train_images = load_mnist(N_train, 'MNIST/train_images.bin')
    train_labels = load_mnist_labels(N_train, 'MNIST/train_labels.bin')
    test_images = load_mnist(1000, 'MNIST/test_images.bin')
    test_labels = load_mnist_labels(1000, 'MNIST/test_labels.bin')
    
    pred_list = []
    test_list = []
    start_time_KNN=time.time()
    for i in range(Number_of_tests-1):
        test_number = random.randint(0, len(test_images)-1)
        pred_label = k_nearest_neighbor(test_images[i],train_images,train_labels,k)
        print(f'tested: {test_labels[i]}')
        print(f'predicted: {pred_label}')
        print(f'index: {i}')
        pred_list.append(pred_label)
        test_list.append(test_labels[i])
    cm = confusion_matrix(test_list, pred_list)
    print(cm)
    error = find_error(cm, Number_of_tests, 10)
    print(error)
    finish_time_KNN=time.time()
    print('Time knn: ', finish_time_KNN-start_time_KNN)
    display_img(test_images[839])
    plot_conf_matrix(cm, error)

    

def RUN_CLUSTERING():
    N_train = 60000
    N_clusters = 64
    train_images = load_mnist(N_train, 'MNIST/train_images.bin')
    train_labels = load_mnist_labels(N_train, 'MNIST/train_labels.bin')
    clust_data = data_for_clustering(train_images, train_labels)
    clustered_templates = get_new_clustered_images(clust_data, N_clusters)
    clustered_label = get_new_clustered_labels(N_clusters, 10)
    
    np.save('MNIST/clustered_templates.npy', clustered_templates)
    np.save('MNIST/clustered_labels.npy', clustered_label)
    
    
def RUN_LABEL_ONE_NN_WITH_CLUSTERING():
    N_test = 1000
    train_images = np.load('MNIST/clustered_templates.npy')
    train_labels = np.load('MNIST/clustered_labels.npy')
    N_train = len(train_labels)
    test_number = random.randint(0,N_test)
    # train_images = load_mnist(N_train, 'MNIST/train_images.bin')
    # train_labels = load_mnist_labels(N_train, 'MNIST/train_labels.bin')
    test_image = load_mnist(N_test, 'MNIST/test_images.bin')[test_number]
    test_label = load_mnist_labels(N_test, 'MNIST/test_labels.bin')[test_number]
    pred_label, pred_image, dummy_vec = nearest_neighbor_scipy(test_image, train_images, train_labels)
    print(f'tested: {test_label}')
    print(f'predicted: {pred_label}')
    display_img(test_image)
    display_img(pred_image.reshape(28,28))


def RUN_LABEL_MULTIPLE_IMAGES_WITH_CLUSTERING():
    Number_of_tests = 50
    # N_train = 2000
    N_test = 1000
    
    train_images = np.load('MNIST/clustered_templates.npy')
    train_labels = np.load('MNIST/clustered_labels.npy')
    print('trainlabels',len(train_labels))
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
    
    
def main():

    RUN_LABEL_MULTIPLE_IMAGES_NN()

    # RUN_LABEL_ONE_KNN()


main()