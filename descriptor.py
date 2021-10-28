import numpy as np
import numpy.linalg as la


def get_m_theta(dx, dy):
    m = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dx, dy)
    return m, theta


def get_patch(image, width, x, y):
    y1 = np.clip(y - width // 2, 0, image.shape[1])
    y2 = np.clip(y + width // 2, 0, image.shape[1])
    x1 = np.clip(x - width // 2, 0, image.shape[0])
    x2 = np.clip(x + width // 2, 0, image.shape[0])
    patch = image[x1:x2, y1:y2]
    return patch


def get_region_hist(m, theta, bins, bin_width, angle, region):
    hist = np.zeros(bins)
    center = bin_width / 2
    for i, (magnitude, t) in enumerate(zip(m.flatten(), theta.flatten())):
        t = (360 + t - angle) % 360
        bin = int(np.floor(t) / bin_width)
        weight = 1 - abs(t - (bin * bin_width + bin_width / 2)) / (bin_width / 2)
        result = magnitude * max(weight, 1e-6)
        x = i // region
        y = i % region
        x_weight = max(1 - abs(x - center) / center, 1e-6)
        y_weight = max(1 - abs(y - center) / center, 1e-6)
        result *= x_weight * y_weight
        hist[bin] += result
    return hist


def normalize_vector(vector):
    norm = la.norm(vector)
    if norm == 0:
        norm = 1e-6
    vector = vector / norm
    return vector


def get_local_descriptors(
        keypoints,
        images,
        differences,
        width=16,
        regions=4,
        bins=8
):
    bin_width = 360 // width
    descriptors = []
    for point in keypoints:
        x, y, p, o = list(map(int, point[:4]))
        angle = point[4]
        image = images[o][p]

        features = np.zeros(bins*regions*regions)
        patch = get_patch(image, width, x, y)
        if not np.min(patch.shape):
            continue
        dx = np.gradient(patch, axis=0)
        dy = np.gradient(patch, axis=1)

        m, theta = get_m_theta(dx, dy)

        region_width = width // regions

        for i in range(regions):
            for j in range(regions):
                x1 = i * region_width
                y1 = j * region_width
                x2 = min(image.shape[0], (i + 1) * region_width)
                y2 = min(image.shape[1], (j + 1) * region_width)
                hist = get_region_hist(
                    m[x1:x2, y1:y2],
                    theta[x1:x2, y1:y2],
                    bins,
                    bin_width,
                    angle,
                    region_width
                )
                from_point = (i * regions + j) * bins
                to_point = (i * regions + j + 1) * bins
                features[from_point:to_point] = hist
        features = normalize_vector(features)
        features[features > 0.2] = 0.2
        features = normalize_vector(features)
        descriptors.append(features)
    return np.array(descriptors)