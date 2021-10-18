import cv2
import numpy as np


def save_images(image_list, prefix):
    for i, element in enumerate(image_list):
        print(f"Save #{i + 1}")
        cv2.imwrite(f"images/{prefix}_{i}.png", element)
