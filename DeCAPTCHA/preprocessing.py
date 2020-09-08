import cv2
import os
import numpy as np
from scipy import ndimage


def segment_image(img):
    images = []

    (dim1, dim2) = img.shape
    r = img.sum(axis=0)

    threshold = dim1 * 255
    init_ind = -1
    for i in range(dim2):
        if r[i] < threshold:
            if init_ind == -1:
                init_ind = i
        if init_ind != -1 and r[i] == threshold and r[i - 1] < threshold:
            tmp = img[:, init_ind:i]

            s = tmp.sum(axis=1)
            th = 255 * (i - init_ind)
            r_i = -1
            for i in range(dim1):
                if s[i] < th and r_i == -1:
                    r_i = i
                if r_i != -1 and s[i] == th and s[i - 1] < th:
                    tmp = tmp[r_i:i, :]
                    break

            init_ind = -1
            tmp = cv2.resize(tmp, (64, 64))
            images.append(tmp)

    return images


def show_image(img):
    cv2.imshow("mat", img)
    cv2.waitKey()


def denoise_img(img, kernel=np.ones((5, 5), np.float), sigma=0.5, erosion_iterations=1):
    v = img

    v = cv2.erode(v, kernel, iterations=erosion_iterations)

    img_gray = ndimage.gaussian_filter(v, sigma)

    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh = cv2.dilate(thresh, kernel, iterations=erosion_iterations)

    return thresh


def load_images_from_folder(folder):
    kernel = np.ones((5, 5), np.float)

    images = []
    labels = []
    threshes = []
    r = 0
    for filename in os.listdir(folder):
        r = r + 1
        if r > 2000:
            exit(0)
        img = cv2.imread(os.path.join(folder, filename))

        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        h, s, v = cv2.split(img)

        thresh = denoise_img(v, kernel, 0.5, 1)

        size = len(filename)
        labels.append(filename[0:size - 4])

        xxc = segment_image(thresh)
        cc = -1
        for i in xxc:
            # print(i.shape)
            cc = cc + 1
            cv2.imwrite("featureimg/" + filename[cc] + str(r) + str(cc) + ".png", i)

        if img is not None:
            threshes.append(thresh)
            images.append(img)
    return images, labels, threshes


# main fn##################################################################
if __name__ == '__main__':
    images = []
    labels = []

    images, labels, threshes = load_images_from_folder("train")

    count = 0
    for filename in os.listdir("train"):
        count += len(filename) - 4
    print(count)
