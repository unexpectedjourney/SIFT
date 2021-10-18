import numpy as np


def is_extremum(value, layer_one, layer_two, layer_three):
    if value > 0 and np.all(layer_one <= value) and \
            np.all(layer_two <= value) and np.all(layer_three <= value):
        return True
    if value < 0 and np.all(layer_one >= value) and \
            np.all(layer_two >= value) and np.all(layer_three >= value):
        return True
    return False


def find_extremum(octaves, sigma, k):
    keypoints = []
    for octave_number, layers in enumerate(octaves):
        for position in range(1, len(layers) - 1):
            for i in range(1, layers[position].shape[0] - 1):
                for j in range(1, layers[position].shape[1] - 1):
                    is_suitable = is_extremum(
                        layers[position][i, j],
                        layers[position - 1][i-1:i+2, j-1:j+2],
                        layers[position][i-1:i+2, j-1:j+2],
                        layers[position + 1][i-1:i+2, j-1:j+2],
                    )
                    if is_suitable:
                        keypoints.append((i, j))
    return keypoints
