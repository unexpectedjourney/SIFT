import cv2
import numpy as np

from utils import resize_by_2


def get_gaussian_kernel(x_size, y_size, sigma):
    x = range(-(x_size - 1) // 2, (x_size - 1) // 2 + 1)
    y = range(-(y_size - 1) // 2, (y_size - 1) // 2 + 1)
    m1, m2 = np.meshgrid(x, y)
    kernel = np.exp(-(m1**2 + m2**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    kernel = kernel / np.sum(kernel)
    return kernel


def conv2d(image, weight):
    output = np.zeros(image.shape)
    x_size = weight.shape[0] // 2
    y_size = weight.shape[1] // 2
    image = np.pad(image, ((y_size,), (x_size,)), mode="constant")
    for i in range(x_size, image.shape[0] - x_size):
        for j in range(y_size, image.shape[1] - y_size):
            patch = image[i-x_size:i+x_size+1, j-y_size:j+y_size+1]
            output[i - x_size, j - y_size] = np.sum(patch * weight)
    return output


def gaussian_filter(image, sigma, truncate=3.):
    width = 2 * int(sigma * truncate + 0.5) + 1
    kernel = get_gaussian_kernel(width, width, sigma)
    output = conv2d(image, kernel)
    return output


def difference_of_images(first_image, second_image):
    return first_image - second_image


def compute_octave(image, sigma, k, rounds=5, first_round=1, verbose=False):
    gaussians = []
    for i in range(first_round, rounds + first_round):
        if verbose:
            print(f"Blur #{i - first_round + 1}")
        sigma_k = sigma * (k ** i)
        blurred = gaussian_filter(image, sigma_k)
        gaussians.append(blurred)

    differences = []
    for i in range(1, len(gaussians)):
        differences.append(difference_of_images(gaussians[i], gaussians[i-1]))

    return gaussians, differences


def generate_octave_pyramid(image, sigma, k, verbose=False):
    octaves_number = int(round(np.log2(np.min(image.shape)) - 1))
    octave_gaussians = []
    octave_differences = []
    for i in range(1, octaves_number):
        if verbose:
            print(f"Pyramid #{i}, shape={image.shape}")
        gaussians, differences = compute_octave(
            image,
            sigma,
            k,
            first_round=i,
            verbose=verbose
        )
        octave_gaussians.append(gaussians)
        octave_differences.append(differences)
        image = resize_by_2(image)
    return octave_gaussians, octave_differences


def main():
    image = cv2.imread("image.jpeg", 0)
    sigma = 1
    k = np.sqrt(2)
    generate_octave_pyramid(image, sigma, k)


if __name__ == "__main__":
    main()
