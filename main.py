import cv2
import numpy as np

from blur import generate_octave_pyramid
from keypoints import find_extremum


def main():
    image = cv2.imread("image.jpeg", 0)
    sigma = 1
    k = np.sqrt(2)
    octave_results = generate_octave_pyramid(image, sigma, k)
    keypoints = find_extremum(octave_results, sigma, k)


if __name__ == "__main__":
    main()
