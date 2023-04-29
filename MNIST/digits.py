import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from statistics import mode, StatisticsError
from collections import Counter
from PIL import Image


from scipy.spatial import distance


# TASK 2: DIGITS

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

# --------------------------------- #
#            LOAD DATA 
# --------------------------------- #

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
    # print('labelveclen',len(label_vec))
    for i in range(len(ref_images)):
        dist_matrix = euclidian_matrix(test_img,ref_images[i])
        dist = true_distance(dist_matrix)
        label_vec[i]= dist
    label = ref_label[np.argmin(label_vec)]
    return label, ref_images[np.argmin(label_vec)], label_vec


def k_nearest_neighbor(test_img, train_images, train_label, k):
    k_nearest = []
    label, ref_image, dist_vec = nearest_neighbor(test_img, train_images, train_label)
    display_img(test_img)
    print(test_img)
    for i in range(k):
        min_index = np.argmin(dist_vec)
        k_nearest.append(train_label[min_index])
        dist_vec[min_index] = 100000000
        display_img(train_images[min_index])
        print(train_images[min_index])
    print(k_nearest)
    pred_label = most_common(k_nearest)
    return pred_label
        
         
# DENNE FUNSKJONEN ER LITT DUST MEN FUNKER
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
        
    print('CLASS COUNT: ', class_count)
    # print('first sort',np.shape(sorted_images_in_rows))
    
    return sorted_images_in_rows
        

def cluster_class(class_images):
    kmeans = KMeans(n_clusters=64)#, random_state=0)

    kmeans.fit(class_images)
    
    clustered_centers = kmeans.cluster_centers_
    
    return clustered_centers


def row_to_image(row):
    local_image=np.zeros((28,28))
    for i in range(28):
        for j in range(28):
            local_image[i][j]=int(row[i*28+j])
    return local_image
    
       
def get_new_clustered_images(sorted_images_in_rows, N_cluster):
    train_images_clustered=np.zeros((N_cluster*10,28,28))
    train_labels_clustered=[]
    for i in range(10):
        # for k in range(len(sorted_images_in_rows[i])):
        #     # print('her', sorted_images_in_rows)
        #     # print('her', sorted_images_in_rows[i])
        #     # print(i)
        #     # sample=sorted_images_in_rows[i]
        #     # sample2= sample[i]
        #     # print(sample2)
        #     # print(np.shape(sample))
            
        clustered_class = cluster_class(sorted_images_in_rows[i])
        for j in range(N_cluster):
            image_local=row_to_image(clustered_class[j])
            train_images_clustered[i*N_cluster+j] = image_local
            # train_images_clustered[i]=cluster_class(sorted_images_in_rows[i])
    return train_images_clustered
        

# def get_labels_BEFORE_CLUSTERING(sorted_images_in_rows):  
#     train_labels_clustered_list = []
#     for i in range(10):
#         num_images_for_class = len(sorted_images_in_rows[i])
#         train_labels_clustered_list.append(np.full(num_images_for_class,i))
#         np.array(train_labels_clustered_list)
#     return train_labels_clustered_list

def get_new_clustered_labels(N_cluster):
    train_labels_clustered_list = []
    for i in range(10):
        for j in range(N_cluster):
            train_labels_clustered_list.append(i)
    train_labels_clustered_list=np.array(train_labels_clustered_list)
    print(train_labels_clustered_list)
    np.save('MNIST/clustered_labels.npy', train_labels_clustered_list)
    return train_labels_clustered_list

def int_cluster(clusterimgs):
    for i in range(640):
        clusterimgs[i]=np.asarray(clusterimgs[i], dtype = 'int')
    return clusterimgs
    
    
# -----------------------------
##########################
# KOOOOOOOOOOOK

def get_clusters(dataset, labels, classes, n_clusters=64):
    label_and_clusters = [] # [ (label, clusters=[]) ]
    for curr_class in classes:
        indices = np.where(labels == curr_class)[0]
        data_of_curr_class = dataset[indices]
        print(np.shape(dataset))
        print(np.shape(data_of_curr_class))

        clusters = KMeans(n_clusters=n_clusters).fit(data_of_curr_class).cluster_centers_

        label_and_clusters.append( (curr_class, clusters) )

    return label_and_clusters

def get_cluster_dataset(dataset, labels, n_clusters=64):
    classes = range(10)
    n_classes = len(classes)
    label_and_clusters = get_clusters(dataset, labels, classes, n_clusters)

    train_data = []
    train_labels = [ [curr_class]*n_clusters for curr_class in classes ]
    train_labels = np.array(train_labels).reshape(n_clusters*n_classes)
    for (label, clusters) in label_and_clusters:
        train_data.extend(clusters)

    return (train_data, train_labels)

def k_nearest_neighbours(data, data_labels, image, k_neighbours):
    dist_and_label_list = []
    for i in range (len(data)):
        dist = distance.euclidean(data[i], image)
        dist_and_label_list.append((dist, data_labels[i]))
    dist_and_label_list = sorted(dist_and_label_list, key=lambda dist: dist[0])

    labels = [ dist_and_label[1] for dist_and_label in dist_and_label_list[:k_neighbours] ]
    return labels

def classify_img(nearest):
    nearest_number_counts = [0]*10
    highest = 0
    number = -1
    for i in range(len(nearest)):
        index = nearest[i]
        nearest_number_counts[index] += 1
    for i in range(len(nearest_number_counts)):
        if nearest_number_counts[i] > highest:
            number = i
    return number

###########################3    FUNNKERRR

def RUN_classify():
    test_predicted = []
    N_train = 1000
    N_test = 10
    
    # train_images = load_mnist(N_train, 'MNIST/train_images.bin')
    # train_labels = load_mnist_labels(N_train, 'MNIST/train_labels.bin')
    
    # train_cluster, train_labels = get_cluster_dataset(train_images.reshape(N_train,784), train_labels, 64)
    # print(np.shape(train_cluster))
    
    train_cluster = np.load('MNIST/clustered_templates.npy')
    train_labels = np.load('MNIST/clustered_labels.npy')

    test_images = load_mnist(N_test, 'MNIST/test_images.bin')
    test_labels = load_mnist_labels(N_test, 'MNIST/test_labels.bin')
    test_images = test_images.reshape(N_test,784)
    train_cluster = np.array(train_cluster).reshape(640,784)
    
    for i in range(N_test):
        k_nearest = k_nearest_neighbours(train_cluster, train_labels, test_images[i], 1)
        predicted_number = classify_img(k_nearest)
        test_predicted.append(predicted_number)
        
        print('\npred', predicted_number)
        print('fasit', test_labels[i])
        
RUN_classify()
    
# #########################################
#------------------------------    

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
    print('len train', len(train_images))
    train_labels = np.load('MNIST/clustered_labels.npy')
    print(train_labels)
    test_image = load_mnist(N_test, 'MNIST/test_images.bin')[test_number]
    test_label = load_mnist_labels(N_test, 'MNIST/test_labels.bin')[test_number]
    pred_label = k_nearest_neighbor(test_image,train_images,train_labels,k)
    print(f'tested: {test_label}')
    print(f'predicted: {pred_label}')

# PRINTING CONFUSION MATRIX OG ERROR RATE USING NN WITH RANDOM TESTING 
def RUN_LABEL_MULTIPLE_IMAGES():
    Number_of_tests = 50
    N_train = 20000
    N_test = 1000
    
    train_images = load_mnist(N_train, 'MNIST/train_images.bin')
    train_labels = load_mnist_labels(N_train, 'MNIST/train_labels.bin')
    test_images = load_mnist(N_test, 'MNIST/test_images.bin')
    test_labels = load_mnist_labels(N_test, 'MNIST/test_labels.bin')
    
    # print(train_images)
    # train_images = np.load('MNIST/clustered_templates.npy')
    # train_images = int_cluster(train_images)
    # print('len train', len(train_images))
    # train_labels = np.load('MNIST/clustered_labels.npy')
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

def RUN_CLUSTERING():
    N_train = 60000
    N_clusters = 64
    train_images = load_mnist(N_train, 'MNIST/train_images.bin')
    train_labels = load_mnist_labels(N_train, 'MNIST/train_labels.bin')
    # display_img(clustered_image)
    clust_data = data_for_clustering(train_images, train_labels)
    clustered_templates = get_new_clustered_images(clust_data, N_clusters)
    clustered_label = get_new_clustered_labels(N_clusters)
    
    np.save('MNIST/clustered_templates.npy', clustered_templates)
    np.save('MNIST/clustered_labels.npy', clustered_label)
    
    loaded = np.load('MNIST/clustered_templates.npy')
    display_img(loaded[128])
    
    # display_img(clustered_templates[128])
    
def RUN_DISPLAY_CLUSTERED_SINGLE_IMAGE():
    N_train = 1000
    train_images = load_mnist(N_train, 'MNIST/train_images.bin')
    clustered_images = np.load('MNIST/clustered_templates.npy')
    display_img(train_images[56])
    display_img(clustered_images[345])
    
def RUN_DISPLAY_CLUSTERED_DIGIT(digit_to_display):
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
    
def RUN_LABEL_ONE_NN_WITH_CLUSTERING():
    N_test = 1000
    train_images = np.load('MNIST/clustered_templates.npy')
    train_labels = np.load('MNIST/clustered_labels.npy')
    print(train_labels)
    N_train = len(train_labels)
    test_number = random.randint(0,N_test)
    # train_images = load_mnist(N_train, 'MNIST/train_images.bin')
    # train_labels = load_mnist_labels(N_train, 'MNIST/train_labels.bin')
    test_image = load_mnist(N_test, 'MNIST/test_images.bin')[test_number]
    test_label = load_mnist_labels(N_test, 'MNIST/test_labels.bin')[test_number]
    pred_label, pred_image, dummy_vec = nearest_neighbor(test_image, train_images, train_labels)
    print(f'tested: {test_label}')
    print(f'predicted: {pred_label}')
    display_img(test_image)
    display_img(pred_image)

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
    
def RUN_PLOT_CONF_MATRIX():
    Number_of_tests = 200
    N_train = 4000
    N_test = 1000
    train_images = load_mnist(N_train, 'MNIST/train_images.bin')
    train_labels = load_mnist_labels(N_train, 'MNIST/train_labels.bin')
    test_images = load_mnist(N_test, 'MNIST/test_images.bin')
    test_labels = load_mnist_labels(N_test, 'MNIST/test_labels.bin')
    pred_list = []
    test_list = []
    for i in range(Number_of_tests):
        test_number = i
        pred_list.append(label_one(train_images, train_labels, test_images[test_number], test_labels[test_number]))
        test_list.append(test_labels[test_number])
    cm = confusion_matrix(test_list, pred_list)
    np.save('MNIST/conf_matrix.npy', cm)
    error = find_error(cm, Number_of_tests, 10)

    # create heatmap
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix with error rate: {round(error,2)*100}%. Number of training data: {N_train}')
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
    
def RUN_TEST_CLUSTERS():
    N_test=640
    train_images = np.load('MNIST/clustered_templates.npy')
    train_images=int_cluster(train_images)
    train_labels = np.load('MNIST/clustered_labels.npy')
    test_number = random.randint(0,N_test)
    index=train_labels[test_number]
    print(index)
    display_img(train_images[test_number])
    display_img(train_images[65])

    print(train_images[65])
    
    
# RUN_TEST_CLUSTERS()

RUN_LABEL_MULTIPLE_IMAGES()

# import_custom_BMP('Test9tall.bmp')
# RUN_PLOT_CONF_MATRIX()
# get_new_clustered_labels(64)

# RUN_LABEL_ONE_NN_WITH_CLUSTERING()

# RUN_LABEL_MULTIPLE_IMAGES_WITH_CLUSTERING()
# RUN_DISPLAY_CLUSTERED_DIGIT(0)

# RUN_LABEL_ONE_KNN()