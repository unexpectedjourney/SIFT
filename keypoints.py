import numpy as np
import numpy.linalg as la

from utils import show_points


def is_extremum(value, layer_one, layer_two, layer_three, threshold):
    t = np.floor(0.5 * threshold / 3 * 255)
    if abs(value) < t:
        return False
    if value > 0 and np.all(layer_one <= value) and \
            np.all(layer_two <= value) and np.all(layer_three <= value):
        return True
    if value < 0 and np.all(layer_one >= value) and \
            np.all(layer_two >= value) and np.all(layer_three >= value):
        return True
    return False


def compute_jacobian(layers, position, x, y):
    d_x = float(layers[position][x + 1, y] - layers[position][x - 1, y]) / 2
    d_y = float(layers[position][x, y + 1] - layers[position][x, y - 1]) / 2
    d_sigma = float(layers[position + 1][x, y] - layers[position - 1][x, y]) / 2
    return np.array((d_x, d_y, d_sigma))


def compute_hessian(layers, position, x, y):
    d_x_x = layers[position][x + 1, y] - 2 * layers[position][x, y] + \
        layers[position][x - 1, y]
    d_x_y = float(
        layers[position][x + 1, y + 1] - layers[position][x - 1, y + 1] -
        layers[position][x + 1, y - 1] + layers[position][x - 1, y - 1]
    ) / 4
    d_x_sigma = float(
        layers[position + 1][x + 1, y] - layers[position + 1][x - 1, y] -
        layers[position - 1][x + 1, y] + layers[position - 1][x - 1, y]
    ) / 4
    d_y_y = layers[position][x, y + 1] - 2 * layers[position][x, y] + \
        layers[position][x, y - 1]
    d_y_sigma = float(
        layers[position + 1][x, y + 1] - layers[position + 1][x, y - 1] -
        layers[position - 1][x, y + 1] + layers[position - 1][x, y - 1]
    ) / 4
    d_sigma_sigma = layers[position + 1][x, y] - 2 * layers[position][x, y] + \
        layers[position - 1][x, y]

    matrix = np.array([
        [d_x_x, d_x_y, d_x_sigma],
        [d_x_y, d_y_y, d_y_sigma],
        [d_x_sigma, d_y_sigma, d_sigma_sigma]
    ])
    return matrix


def compute_subpixel(layers, position, x, y):
    jacobian = compute_jacobian(layers, position, x, y)
    hessian = compute_hessian(layers, position, x, y)
    x_hat = -la.inv(hessian) @ jacobian

    return x_hat, hessian, jacobian


def get_contrast(value, x_hat, jacobian):
    return value + jacobian @ x_hat * 0.5


def get_harris_value(hessian):
    a = (hessian[0, 0] + hessian[1, 1]) ** 2
    b = hessian[0, 0] * hessian[1, 1] - hessian[0, 1] ** 2
    return a / b if b != 0 else 0


def find_extremum(octaves, sigma, k, contrast_threshold=0.4, edge_threshold=10):
    keypoints = []
    initial_keypoints = 0
    contrast_keypoints = 0
    harris_keypoints = 0
    for octave_number, layers in enumerate(octaves):
        local_keypoints = []
        for position in range(1, len(layers) - 1):
            for i in range(1, layers[position].shape[0] - 1):
                for j in range(1, layers[position].shape[1] - 1):
                    is_suitable = is_extremum(
                        layers[position][i, j],
                        layers[position - 1][i-1:i+2, j-1:j+2],
                        layers[position][i-1:i+2, j-1:j+2],
                        layers[position + 1][i-1:i+2, j-1:j+2],
                        contrast_threshold,
                    )
                    if not is_suitable:
                        continue
                    initial_keypoints += 1

                    try:
                        x_hat, hessian, jacobian = compute_subpixel(
                            layers,
                            position,
                            i,
                            j
                        )
                    except la.LinAlgError:
                        continue
                    contrast = get_contrast(
                        layers[position][i, j],
                        x_hat,
                        jacobian
                    )
                    if np.abs(contrast) < contrast_threshold:
                        continue
                    contrast_keypoints += 1

                    edge_value = get_harris_value(hessian)
                    if edge_value > edge_threshold:
                        continue
                    harris_keypoints += 1
                    keypoint = np.array([i, j, position]) + x_hat
                    local_keypoints.append(keypoint)
        show_points(layers, local_keypoints, octave_number)
        keypoints.extend(local_keypoints)
    print(initial_keypoints, contrast_keypoints, harris_keypoints)
    return keypoints
