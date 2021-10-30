import cv2
import numpy as np

from blur import generate_octave_pyramid
from keypoints import find_extremum, get_oriented_keypoints
from descriptor import get_local_descriptors


def main():
    image = cv2.imread("image.jpeg", 0)
    sigma = 1
    k = np.sqrt(2)
    octave_gaussians, octave_differences = generate_octave_pyramid(image, sigma, k)
    keypoints = find_extremum(octave_differences, sigma, k)
    keypoints = get_oriented_keypoints(
        keypoints,
        octave_gaussians,
        octave_differences
    )
    print(len(keypoints))
    get_local_descriptors(keypoints, octave_gaussians)


if __name__ == "__main__":
    main()
