import numpy as np
import numpy.linalg as la

from blur import get_gaussian_kernel


def is_extremum(value, layer_one, layer_two, layer_three):
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


def get_m_theta(image, x, y):
    x_p = np.clip(x+1, 0, image.shape[0]-1)
    x_m = np.clip(x-1, 0, image.shape[0]-1)
    y_p = np.clip(y+1, 0, image.shape[1]-1)
    y_m = np.clip(y-1, 0, image.shape[1]-1)

    d_x = image[x_p, y] - image[x_m, y]
    d_y = image[x, y_p] - image[x, y_m]
    m = np.sqrt(d_x ** 2 + d_y ** 2)
    theta = np.arctan2(d_x, d_y)
    return m, theta


def fit_parabola(hist, index, width):
    center_point = index * width + width // 2
    right_point = ((index+1) * width + width // 2) % 360
    left_point = (360 + (index-1) * width + width // 2) % 360

    A = np.array([
        [center_point**2, center_point, 1],
        [right_point**2, right_point, 1],
        [left_point**2, left_point, 1]])
    b = np.array([
        hist[index],
        hist[(index+1) % hist.shape[0]],
        hist[(index-1) % hist.shape[0]]])
    x = la.lstsq(A, b, rcond=None)[0]

    value = -x[1] / (2*x[0] + 1e-6)
    return value


def compute_keypoint_hist(image, x, y, bin_width, p=1, bins=36):
    sigma = p * 1.5
    width = 2 * int(round(sigma)) + 1
    if width % 2 == 0:
        width += 1

    kernel = get_gaussian_kernel(width, width, sigma)
    hist = np.zeros(bins, dtype=float)
    for i in range(-width // 2, width // 2 + 1):
        for j in range(-width // 2, width // 2 + 1):
            xx = np.clip(x + i, 1, image.shape[0] - 2)
            yy = np.clip(y + j, 1, image.shape[1] - 2)

            m, theta = get_m_theta(image, xx, yy)
            weight = kernel[i + width // 2, j + width // 2] * m
            bin = int(np.floor(theta) // bin_width)

            hist[bin] += weight
    return hist


def extract_keypoints_from_hist(hist, point, bin_width):
    new_points = []
    max_bin = np.argmax(hist)
    max_value = hist[max_bin]
    new_points.append(np.array([*point, fit_parabola(hist, max_bin, bin_width)]))

    for i, value in enumerate(hist):
        if .8 * max_value <= value and i != max_bin:
            new_points.append(np.array([*point, fit_parabola(hist, i, bin_width)]))

    return new_points


def get_oriented_keypoints(keypoints, images, differences, bins=36):
    new_points = []
    bin_width = 360 // bins

    for point in keypoints:
        x, y, p, o = point
        image = images[o][p]
        hist = compute_keypoint_hist(
            image,
            x,
            y,
            bin_width=bin_width,
            p=p,
            bins=bins,
        )
        new_points.extend(extract_keypoints_from_hist(hist, point, bin_width))
    return np.array(new_points)


def find_extremum(difference_octaves, sigma, k, contrast_threshold=0.4, edge_threshold=10):
    keypoints = []
    initial_keypoints = 0
    contrast_keypoints = 0
    harris_keypoints = 0
    for octave_number, layers in enumerate(difference_octaves):
        local_keypoints = []
        for position in range(1, len(layers) - 1):
            for i in range(1, layers[position].shape[0] - 1):
                for j in range(1, layers[position].shape[1] - 1):
                    is_suitable = is_extremum(
                        layers[position][i, j],
                        layers[position - 1][i-1:i+2, j-1:j+2],
                        layers[position][i-1:i+2, j-1:j+2],
                        layers[position + 1][i-1:i+2, j-1:j+2],
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
                    keypoint[0] = keypoint[0] * 2**octave_number
                    keypoint[1] = keypoint[1] * 2**octave_number
                    keypoint = np.array([*keypoint, octave_number])
                    local_keypoints.append(keypoint)
        keypoints.extend(local_keypoints)
    return keypoints
