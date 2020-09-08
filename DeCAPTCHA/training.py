import numpy as np
import os
import cv2
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pickle
import time


def test_data(model, X, y=None, analyse=True):
    t0 = time.time()
    predicted_labels = model.predict(X)
    t1 = time.time()

    if analyse:
        if y is None:
            print("Error")
        print("Accuracy : ", model.score(X, y))
        print("Prediction Time ", t1 - t0)


def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)

        size = len(filename)
        labels.append(ord(filename[0:1]) - 65)

        if img is not None:
            img = np.array(img)

            images.append(img.reshape(64 * 64))
    return images, labels


images = []
labels = []

images, labels = load_images_from_folder("featureimg")


def svm(images, labels):
    clf = LinearSVC(C=5)
    data_train, data_test, label_train, label_test = train_test_split(images, labels, train_size=0.9, test_size=0.1)
    tic = time.time()
    clf.fit(data_train, label_train)

    toc = time.time()
    print("Training Time is : ", toc - tic)
    pickle.dump(clf, open("OVA_CSVM", "wb"))
    clf = pickle.load(open("OVA_CSVM", "rb"))
    test_data(clf, data_test, label_test)
    # print(cross_val_score(clf,images,labels,cv=3,n_jobs=-1))


##################
svm(images, labels)
