import os
# import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.decomposition import PCA


def apply_PCA(n, training_set, testing_set): 
# def apply_PCA(n, evectors, evalues, temp, mean_training_set): 
    temp = np.zeros((training_set.shape[0], height * width))
    mean_training_set = np.sum(training_set, axis=0) / training_set.shape[0]
    plt.subplot(151)
    plt.imshow(mean_training_set.reshape(56, 46), cmap='Greys_r')
    plt.title("Mean face")

    for i in range (0, training_set.shape[0]):
        temp[i] = training_set[i] - mean_training_set # temp[i] = Xi-u
    C = np.matrix(temp.transpose()) * np.matrix(temp) 
    C /= training_set.shape[0]

    evalues, evectors = np.linalg.eig(C)                          # eigenvectors/values of the covariance matrix
    sort_indices = evalues.argsort()[::-1]                        # getting decreasing order
    evalues = evalues[sort_indices]                               # putting evalues and evectors in that order
    evectors = evectors[:,sort_indices]                           # eigenvector is in column
    plt.subplot(152)
    plt.imshow(evectors[:, 0].real.reshape(56, 46), cmap='Greys_r')
    plt.title("First eigenfaces")
    plt.subplot(153)
    plt.imshow(evectors[:, 1].real.reshape(56, 46), cmap='Greys_r')
    plt.title("Second eigenfaces")
    plt.subplot(154)
    plt.imshow(evectors[:, 2].real.reshape(56, 46), cmap='Greys_r')
    plt.title("Third eigenfaces")
    plt.subplot(155)
    plt.imshow(evectors[:, 3].real.reshape(56, 46), cmap='Greys_r')
    plt.title("Fourth eigenfaces")

    # plt.show()

    # taking only the top n components
    correspondingEigenValues  = evalues[:n]
    correspondingEigenVectors = evectors[:,:n]
    # projection of the data
    correspondingProjection = np.dot(temp, correspondingEigenVectors)
    constructedImages = np.dot(correspondingProjection, correspondingEigenVectors.transpose()) + mean_training_set
    return constructedImages

def MSE_loss(compress_img, img): 
    mse_loss = 0
    # print(img.shape)
    for pixel in range(img.shape[0]):
        mse_loss += (compress_img[:, pixel].real - img[pixel]) ** 2
    mse_loss = mse_loss / img.shape[0]
    return float(mse_loss)

faces_dir = './p1_data'
person_num = 40
face_num = 10
width = 46
height = 56
# training_set = np.zeros(person_num * (face_num - 1), height, width)
# testing_set = np.zeros(person_num, height, width)
training_set = []
testing_set = []
training_label = []
testing_label = []

for person_id in range(1, person_num + 1):
    for face_id in range(1, face_num + 1):
        path_to_img = os.path.join(faces_dir,str(person_id) + '_' + str(face_id) + '.png')
        img = mping.imread(path_to_img, 0)
        if face_id == face_num:
            testing_set.append(img.flatten())
            testing_label.append(person_id)
        else:
            training_set.append(img.flatten())
            training_label.append(person_id)
training_set = np.array(training_set)
testing_set = np.array(testing_set)
# print(training_set.shape)  
# print(testing_set.shape)            
# plt.imshow(training_set[0], cmap='Greys_r')
# plt.imshow(testing_set[0], cmap='Greys_r')

# temp = np.zeros((training_set.shape[0], height * width))
# mean_training_set = np.sum(training_set, axis=0) / training_set.shape[0]
# plt.subplot(151)
# plt.imshow(mean_training_set.reshape(56, 46), cmap='Greys_r')
# plt.title("Mean face")

# print(mean_training_set.shape)

# for i in range (0, training_set.shape[0]):
#     temp[i] = training_set[i] - mean_training_set # temp[i] = Xi-u
# C = np.matrix(temp.transpose()) * np.matrix(temp) 
# C /= training_set.shape[0]

# evalues, evectors = np.linalg.eig(C)                          # eigenvectors/values of the covariance matrix
# sort_indices = evalues.argsort()[::-1]                        # getting decreasing order
# evalues = evalues[sort_indices]                               # putting evalues and evectors in that order
# evectors = evectors[:,sort_indices]                           # eigenvector is in column
# plt.subplot(152)
# plt.imshow(evectors[:, 0].real.reshape(56, 46), cmap='Greys_r')
# plt.title("First eigenfaces")
# plt.subplot(153)
# plt.imshow(evectors[:, 1].real.reshape(56, 46), cmap='Greys_r')
# plt.title("Second eigenfaces")
# plt.subplot(154)
# plt.imshow(evectors[:, 2].real.reshape(56, 46), cmap='Greys_r')
# plt.title("Third eigenfaces")
# plt.subplot(155)
# plt.imshow(evectors[:, 3].real.reshape(56, 46), cmap='Greys_r')
# plt.title("Fourth eigenfaces")

# plt.show()


# constructedImages = apply_PCA(3, training_set, testing_set)
# plt.subplot(151)
# plt.imshow(constructedImages[9].real.reshape(56, 46), cmap='Greys_r')
# plt.title("n=3")
# mse_loss = MSE_loss(constructedImages[9], training_set[9])
# print(mse_loss)

# constructedImages = apply_PCA(50, training_set, testing_set)
# plt.subplot(152)
# plt.imshow(constructedImages[9].real.reshape(56, 46), cmap='Greys_r')
# plt.title("n=50")
# mse_loss = MSE_loss(constructedImages[9], training_set[9])
# print(mse_loss)

# constructedImages = apply_PCA(170, training_set, testing_set)
# plt.subplot(153)
# plt.imshow(constructedImages[9].real.reshape(56, 46), cmap='Greys_r')
# plt.title("n=170")
# mse_loss = MSE_loss(constructedImages[9], training_set[9])
# print(mse_loss)

# constructedImages = apply_PCA(240, training_set, testing_set)
# plt.subplot(154)
# plt.imshow(constructedImages[9].real.reshape(56, 46), cmap='Greys_r')
# plt.title("n=240")
# mse_loss = MSE_loss(constructedImages[9], training_set[9])
# print(mse_loss)

# constructedImages = apply_PCA(345, training_set, testing_set)
# plt.subplot(155)
# plt.imshow(constructedImages[9].real.reshape(56, 46), cmap='Greys_r')
# plt.title("n=345")
# mse_loss = MSE_loss(constructedImages[9], training_set[9])
# print(mse_loss)

# plt.show()


k_value = [1, 3, 5]
n_value = [3, 50, 170]

first_fold = []
second_fold = []
third_fold = []
first_fold_label = []
second_fold_label = []
third_fold_label = []

for i in range(training_set.shape[0]):
    if (i // 3) % 3 == 0:
        first_fold.append(training_set[i])
        first_fold_label.append((i // 9) + 1)
    elif (i // 3) % 3 == 1:
        second_fold.append(training_set[i])
        second_fold_label.append((i // 9) + 1)
    else:
        third_fold.append(training_set[i])
        third_fold_label.append((i // 9) + 1)
first_fold = np.array(first_fold)
second_fold = np.array(second_fold)
third_fold = np.array(third_fold)
first_fold_label = np.array(first_fold_label)
second_fold_label = np.array(second_fold_label)
third_fold_label = np.array(third_fold_label)
# print(first_fold_label.shape)
# print(second_fold_label.shape)
# print(third_fold_label.shape)

# print(np.hstack((first_fold_label, second_fold_label)).shape)

# print("Fold 1")
# # Fold 1
# for k in k_value:
#     for n in n_value:
#         KNN = KNeighborsClassifier(n_neighbors=k)
#         constructedImages = apply_PCA(n, np.vstack((second_fold, third_fold)), first_fold)
#         KNN.fit(constructedImages, np.hstack((second_fold_label, third_fold_label)))
#         print("k = " + str(k), ", n = " + str(n),"Accuracy = " + str(KNN.score(first_fold, first_fold_label)))

# print("Fold 2")
# # Fold 2
# for k in k_value:
#     for n in n_value:
#         KNN = KNeighborsClassifier(n_neighbors=k)
#         constructedImages = apply_PCA(n, np.vstack((first_fold, third_fold)), second_fold)
#         KNN.fit(constructedImages, np.hstack((first_fold_label, third_fold_label)))
#         print("k = " + str(k), ", n = " + str(n),"Accuracy = " + str(KNN.score(second_fold, second_fold_label)))

# print("Fold 3")
# # Fold 3
# for k in k_value:
#     for n in n_value:
#         KNN = KNeighborsClassifier(n_neighbors=k)
#         constructedImages = apply_PCA(n, np.vstack((first_fold, second_fold)), third_fold)
#         KNN.fit(constructedImages, np.hstack((first_fold_label, second_fold_label)))
#         print("k = " + str(k), ", n = " + str(n),"Accuracy = " + str(KNN.score(third_fold, third_fold_label)))

print("Testing")
#Testing 
KNN = KNeighborsClassifier(n_neighbors=1)
constructedImages = apply_PCA(50, training_set, testing_set)
KNN.fit(constructedImages, training_label)
print("k = " + str(1), ", n = " + str(50),"Accuracy = " + str(KNN.score(testing_set, testing_label)))

KNN = KNeighborsClassifier(n_neighbors=1)
constructedImages = apply_PCA(170, training_set, testing_set)
KNN.fit(constructedImages, training_label)
print("k = " + str(1), ", n = " + str(170),"Accuracy = " + str(KNN.score(testing_set, testing_label)))





