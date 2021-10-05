import os
# import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

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
        img = mping.imread(path_to_img,0)
        if face_id == face_num:
            testing_set.append(img.flatten())
            testing_label.append(person_id)
        else:
            training_set.append(img.flatten())
            training_label.append(person_id)
training_set = np.array(training_set)
testing_set = np.array(testing_set)

k_value = [1, 3, 5]
n_value = [3, 50, 170]
# Fold 1 n = 1
KNN = KNeighborsClassifier(n_neighbors=1)
KNN.fit(training_set, training_label)
print(KNN.score(testing_set, testing_label))