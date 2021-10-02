import os
# import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
from sklearn.neighbors import KNeighborsClassifier

def apply_PCA(n, evectors, evalues, temp, mean_training_set): 
    # taking only the top n components
    correspondingEigenValues  = evalues[:n]
    correspondingEigenVectors = evectors[:,:n]
    # projection of the data
    correspondingProjection = np.dot(temp, correspondingEigenVectors)
    constructedImages = np.dot(correspondingProjection, correspondingEigenVectors.transpose()) + mean_training_set
    return constructedImages

def MSE_loss(compress_img, img): 
    mse_loss = 0
    print(img.shape)
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
        img = mping.imread(path_to_img)
        if face_id == face_num:
            testing_set.append(img.flatten())
            testing_label.append(person_id)
        else:
            training_set.append(img.flatten())
            training_label.append(person_id)
training_set = np.array(training_set)
testing_set = np.array(testing_set)
print(training_set.shape)  
print(testing_set.shape)            
# plt.imshow(training_set[0], cmap='Greys_r')
# plt.imshow(testing_set[0], cmap='Greys_r')

temp = np.zeros((training_set.shape[0], height * width))
mean_training_set = np.sum(training_set, axis=0) / training_set.shape[0]
plt.subplot(151)
plt.imshow(mean_training_set.reshape(56, 46), cmap='Greys_r')
plt.title("Mean face")

print(mean_training_set.shape)

for i in range (0, training_set.shape[0]):
    temp[i] = training_set[i] - mean_training_set # temp[i] = Xi-u
print(temp[0])
C = np.matrix(temp.transpose()) * np.matrix(temp) 
# C = (temp).dot(temp.T)
C /= training_set.shape[0]
# C = C / training_set.shape[0]
print(C.shape)

evalues, evectors = np.linalg.eig(C)                          # eigenvectors/values of the covariance matrix
sort_indices = evalues.argsort()[::-1]                        # getting decreasing order
# sort_indices = np.argsort(-evalues)                        # getting decreasing order
evalues = evalues[sort_indices]                               # putting evalues and evectors in that order
evectors = evectors[:,sort_indices]                           # eigenvector is in column
# evalues = evalues[:360] 
# evectors = evectors[:360] 
print(C[0])
print(evectors.shape)
# norms = np.linalg.norm(evectors, axis=0) 
# evectors = evectors / norms 
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

plt.show()


constructedImages = apply_PCA(3, evectors, evalues, temp, mean_training_set)
plt.subplot(151)
plt.imshow(constructedImages[9].real.reshape(56, 46), cmap='Greys_r')
plt.title("n=3")

constructedImages = apply_PCA(50, evectors, evalues, temp, mean_training_set)
plt.subplot(152)
plt.imshow(constructedImages[9].real.reshape(56, 46), cmap='Greys_r')
plt.title("n=50")

constructedImages = apply_PCA(170, evectors, evalues, temp, mean_training_set)
plt.subplot(153)
plt.imshow(constructedImages[9].real.reshape(56, 46), cmap='Greys_r')
plt.title("n=170")

constructedImages = apply_PCA(240, evectors, evalues, temp, mean_training_set)
plt.subplot(154)
plt.imshow(constructedImages[9].real.reshape(56, 46), cmap='Greys_r')
plt.title("n=240")

constructedImages = apply_PCA(345, evectors, evalues, temp, mean_training_set)
plt.subplot(155)
plt.imshow(constructedImages[9].real.reshape(56, 46), cmap='Greys_r')
plt.title("n=345")

plt.show()

mse_loss = MSE_loss(constructedImages[9], training_set[9])
print(mse_loss)
# constructedImages = np.dot(proj, evectors.transpose()) + mean_training_set
# plt.subplot(121)
# plt.imshow(constructedImages[0].real.reshape(56, 46), cmap='Greys_r')
# plt.subplot(122)
# plt.imshow(training_set[0].reshape(56, 46), cmap='Greys_r')




