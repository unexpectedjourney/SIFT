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
    return np.array([d_x, d_y, d_sigma])


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


def clip_values(keypoint, layers):
    pp = int(np.round(keypoint[2]))
    pp = np.clip(pp, 1, len(layers) - 2)
    xx = int(np.round(keypoint[0]))
    xx = np.clip(xx, 1, layers[0].shape[0] - 2)
    yy = int(np.round(keypoint[1]))
    yy = np.clip(yy, 1, layers[0].shape[1] - 2)
    new_kp = np.array([xx, yy, pp], dtype=float)
    return new_kp


def compute_subpixel(layers, position, x, y, n_iter=5):
    is_done = False
    keypoint = np.array([x, y, position], dtype=np.float)
    for _ in range(n_iter):
        xx, yy, pp = map(int, keypoint)
        jacobian = compute_jacobian(layers, pp, xx, yy)
        hessian = compute_hessian(layers, pp, xx, yy)
        x_hat = -la.inv(hessian) @ jacobian
        keypoint += x_hat

        keypoint = clip_values(keypoint, layers)
        if np.all(np.abs(x_hat) < 0.5):
            is_done = True
            break
    keypoint = keypoint.astype(int)
    return keypoint, x_hat, hessian, jacobian, is_done


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
                        keypoint, x_hat, hessian, jacobian, is_done = compute_subpixel(
                            layers,
                            position,
                            i,
                            j
                        )
                    except la.LinAlgError as e:
                        print(e)
                        continue

                    if not is_done:
                        continue

                    x = keypoint[0]
                    y = keypoint[1]
                    p = keypoint[2]
                    contrast = get_contrast(
                        layers[p][x, y],
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
                    local_keypoints.append(keypoint)
        show_points(layers, local_keypoints, octave_number)
        keypoints.extend(local_keypoints)
    print(initial_keypoints, contrast_keypoints, harris_keypoints)
    return keypoints
