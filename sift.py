import cv2
import numpy as np

from blur import generate_octave_pyramid
from keypoints import find_extremum, get_oriented_keypoints
from descriptor import get_local_descriptors


class SIFT:
    def __init__(self):
        self.sigma = 1
        self.k = np.sqrt(2)

    def compute(self, image_path):
        image = cv2.imread(image_path, 0)
        octave_gaussians, octave_differences = generate_octave_pyramid(
            image,
            self.sigma,
            self.k
        )
        keypoints = find_extremum(octave_differences, self.sigma, self.k)
        keypoints = get_oriented_keypoints(
            keypoints,
            octave_gaussians,
            octave_differences
        )
        descriptors = get_local_descriptors(keypoints, octave_gaussians)
        return keypoints, descriptors
