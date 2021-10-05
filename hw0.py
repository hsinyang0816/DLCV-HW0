import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
from sklearn.neighbors import KNeighborsClassifier

def apply_PCA(n, training_set):  
    temp = np.zeros((training_set.shape[0], height * width))
    mean_training_set = np.sum(training_set, axis=0) / training_set.shape[0]
    for i in range (0, training_set.shape[0]):
        temp[i] = training_set[i] - mean_training_set # temp[i] = Xi-u
    C = np.matrix(temp.transpose()) * np.matrix(temp) 
    C /= training_set.shape[0]
    evalues, evectors = np.linalg.eig(C)                          # eigenvectors/values of the covariance matrix
    sort_indices = evalues.argsort()[::-1]                        # getting decreasing order
    evalues = evalues[sort_indices]                               # putting evalues and evectors in that order
    evectors = evectors[:,sort_indices]                           # eigenvector is in column
    # taking only the top n components
    correspondingEigenValues  = evalues[:n]
    correspondingEigenVectors = evectors[:,:n]
    # projection of the data
    correspondingProjection = np.dot(temp, correspondingEigenVectors)
    constructedImages = np.dot(correspondingProjection, correspondingEigenVectors.transpose()) + mean_training_set
    return constructedImages, evectors, mean_training_set

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

_, evectors, mean_training_set = apply_PCA(360, training_set)
plt.subplot(151)
plt.imshow(mean_training_set.reshape(56, 46), cmap='Greys_r')
plt.title("Mean face")
num = ["First", "Second", "Third", "Fourth"]
for i in range(2, 6):
    plt.subplot(1, 5, i)
    plt.imshow(evectors[:, i-2].real.reshape(56, 46), cmap='Greys_r')
    plt.title( num[i-2] + " eigenfaces" )

plt.show()
n_sequence = [3, 50, 170, 240, 345]
for i in range(1, 6):
    constructedImages, _, _ = apply_PCA(n_sequence[i-1], training_set)
    plt.subplot(1, 5, i)
    plt.imshow(constructedImages[9].real.reshape(56, 46), cmap='Greys_r')
    plt.title("n=" + str(n_sequence[i-1]))
    mse_loss = MSE_loss(constructedImages[9], training_set[9])
    print("The MSE loss of n = " + str(n_sequence[i-1]) + " : ", mse_loss)

plt.show()

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

print("Fold 1")
# Fold 1
for k in k_value:
    for n in n_value:
        KNN = KNeighborsClassifier(n_neighbors=k)
        constructedImages, _, _ = apply_PCA(n, np.vstack((second_fold, third_fold)))
        KNN.fit(constructedImages, np.hstack((second_fold_label, third_fold_label)))
        validation_construct_image, _, _ = apply_PCA(n, first_fold)
        print("k = " + str(k), ", n = " + str(n),"Accuracy = " + str(KNN.score(validation_construct_image, first_fold_label)))

print("Fold 2")
# Fold 2
for k in k_value:
    for n in n_value:
        KNN = KNeighborsClassifier(n_neighbors=k)
        constructedImages, _, _ = apply_PCA(n, np.vstack((first_fold, third_fold)))
        KNN.fit(constructedImages, np.hstack((first_fold_label, third_fold_label)))
        validation_construct_image, _, _ = apply_PCA(n, second_fold)
        print("k = " + str(k), ", n = " + str(n),"Accuracy = " + str(KNN.score(validation_construct_image, second_fold_label)))

print("Fold 3")
# Fold 3
for k in k_value:
    for n in n_value:
        KNN = KNeighborsClassifier(n_neighbors=k)
        constructedImages, _, _ = apply_PCA(n, np.vstack((first_fold, second_fold)))
        KNN.fit(constructedImages, np.hstack((first_fold_label, second_fold_label)))
        validation_construct_image, _, _ = apply_PCA(n, third_fold)
        print("k = " + str(k), ", n = " + str(n),"Accuracy = " + str(KNN.score(validation_construct_image, third_fold_label)))

print("Testing")
#Testing 
KNN = KNeighborsClassifier(n_neighbors=1)
constructedImages, _, _ = apply_PCA(50, training_set)
KNN.fit(constructedImages, training_label)
testing_construct_image, _, _ = apply_PCA(50, testing_set)
print("k = " + str(1), ", n = " + str(50),"Accuracy = " + str(KNN.score(testing_construct_image, testing_label)))

KNN = KNeighborsClassifier(n_neighbors=1)
constructedImages, _, _ = apply_PCA(170, training_set)
KNN.fit(constructedImages, training_label)
testing_construct_image, _, _ = apply_PCA(170, testing_set)
print("k = " + str(1), ", n = " + str(170),"Accuracy = " + str(KNN.score(testing_construct_image, testing_label)))





