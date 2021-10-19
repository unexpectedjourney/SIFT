import cv2
import numpy as np

from utils import save_images, resize_by_2


def get_gaussian_kernel(x_size, y_size, sigma):
    x = range(-(x_size - 1) // 2, (x_size - 1) // 2 + 1)
    y = range(-(y_size - 1) // 2, (y_size - 1) // 2 + 1)
    m1, m2 = np.meshgrid(x, y)
    kernel = np.exp(-(m1**2 + m2**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    kernel = kernel / np.sum(kernel)
    return kernel


def patch_multiplication(image, weight, x, y):
    result = 0
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            image_i = x + i - weight.shape[0] // 2
            image_j = y + j - weight.shape[1] // 2

            if image_i < 0 or image_j < 0 or image_i >= image.shape[0] or \
                    image_j >= image.shape[1]:
                continue
            result = result + image[image_i, image_j] * weight[i, j]
    return int(result)


def conv2d(image, weight):
    output = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = patch_multiplication(image, weight, i, j)
    return output


def gaussian_filter(image, sigma, truncate=3.):
    width = int(sigma * truncate + 0.5) + 1
    kernel = get_gaussian_kernel(width, width, sigma)
    output = conv2d(image, kernel)
    return output


def difference_of_images(first_image, second_image):
    return first_image - second_image


def compute_octave(image, sigma, k, rounds=5, first_round=1):
    gaussians = []
    for i in range(first_round, rounds + first_round):
        print(f"Blur #{i - first_round + 1}")
        sigma_k = sigma * (k ** i)
        blurred = gaussian_filter(image, sigma_k)
        gaussians.append(blurred)

    save_images(gaussians, f"image_{first_round}")
    differences = []
    for i in range(1, len(gaussians)):
        differences.append(difference_of_images(gaussians[i], gaussians[i-1]))

    return differences


def generate_octave_pyramid(image, sigma, k):
    octaves_number = int(round(np.log2(np.min(image.shape)) - 1))
    octave_results = []
    for i in range(1, octaves_number):
        print(f"Pyramid #{i}, shape={image.shape}")
        differences = compute_octave(image, sigma, k, first_round=i)
        octave_results.append(differences)
        image = resize_by_2(image)
    return octave_results


def main():
    image = cv2.imread("image.jpeg", 0)
    sigma = 1
    k = np.sqrt(2)
    generate_octave_pyramid(image, sigma, k)


if __name__ == "__main__":
    main()
