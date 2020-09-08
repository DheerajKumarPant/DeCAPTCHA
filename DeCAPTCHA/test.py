import os
import numpy as np
import cv2
import pickle
import preprocessing

for pics in ["AGIA","BIS","BIU","HYJ","IAF","IIC","IJU","ISJ","PJK","LIQ","LITS"]:

    y = cv2.imread(pics+".png")
    y = cv2.cvtColor(y, cv2.COLOR_RGB2HSV)
    _, _, y = cv2.split(y)
    preprocessing.show_image(y)

    y = preprocessing.denoise_img(y)

    list_char_imgs = []
    list_char_imgs = preprocessing.segment_image(y)

    clf = pickle.load(open("OVA_CSVM", "rb"))

    for i in list_char_imgs:
        print(chr(65+clf.predict(i.reshape(1, 64*64))[0]))


