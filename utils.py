import cv2
import numpy as np


def save_images(image_list, prefix):
    for i, element in enumerate(image_list):
        print(f"Save #{i + 1}")
        cv2.imwrite(f"images/{prefix}_{i}.png", element)


def resize_by_2(image):
    return image[::2, ::2]


def show_points(layers, keypoints):
    image = cv2.imread("image.jpeg", 0)
    img = image.copy()
    for point in keypoints:
        x = int(np.round(point[0]))
        y = int(np.round(point[1]))
        img = cv2.circle(img, (y, x), 5, (255, 0, 0), 2)

    cv2.imwrite("output.png", img)

    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(image, None)

    img1 = cv2.drawKeypoints(image, keypoints_1, image)

    cv2.imwrite("output1.png", img1)
