import os

import torch
import numpy as np
from tqdm import tqdm

from sift import SIFT

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset_path = 'hpatches-sequences-release'
lim = [1, 15]
rng = np.arange(lim[0], lim[1] + 1)
n_i = 52
n_v = 56
cache_dir = 'cache'
errors = {}
sift = SIFT()

if not os.path.isdir(cache_dir):
    os.mkdir(cache_dir)


def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()


def benchmark_features(read_feats):
    seq_names = sorted(os.listdir(dataset_path))

    n_feats = []
    n_matches = []
    seq_type = []
    i_err = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}

    for seq_idx, seq_name in tqdm(enumerate(seq_names), total=len(seq_names)):
        keypoints_a, descriptors_a = read_feats(seq_name, 1)
        n_feats.append(keypoints_a.shape[0])

        for im_idx in range(2, 7):
            keypoints_b, descriptors_b = read_feats(seq_name, im_idx)
            n_feats.append(keypoints_b.shape[0])

            matches = mnn_matcher(
                torch.from_numpy(descriptors_a).to(device=device),
                torch.from_numpy(descriptors_b).to(device=device)
            )

            homography = np.loadtxt(os.path.join(dataset_path, seq_name, f"H_1_{im_idx}"))

            pos_a = keypoints_a[matches[:, 0], :2]
            pos_a_h = np.concatenate([pos_a, np.ones([matches.shape[0], 1])], axis=1)
            pos_b_proj_h = np.transpose(np.dot(homography, np.transpose(pos_a_h)))
            pos_b_proj = pos_b_proj_h[:, :2] / pos_b_proj_h[:, 2:]

            pos_b = keypoints_b[matches[:, 1], :2]

            dist = np.sqrt(np.sum((pos_b - pos_b_proj) ** 2, axis=1))

            n_matches.append(matches.shape[0])
            seq_type.append(seq_name[0])

            if dist.shape[0] == 0:
                dist = np.array([float("inf")])

            for thr in rng:
                if seq_name[0] == 'i':
                    i_err[thr] += np.mean(dist <= thr)
                else:
                    v_err[thr] += np.mean(dist <= thr)

    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)
    n_matches = np.array(n_matches)

    return i_err, v_err, [seq_type, n_feats, n_matches]


def summary(stats):
    seq_type, n_feats, n_matches = stats
    print('# Features: {:f} - [{:d}, {:d}]'.format(
        np.mean(n_feats), np.min(n_feats), np.max(n_feats))
    )
    print('# Matches: Overall {:f}, Illumination {:f}, Viewpoint {:f}'.format(
        np.sum(n_matches) / ((n_i + n_v) * 5),
        np.sum(n_matches[seq_type == 'i']) / (n_i * 5),
        np.sum(n_matches[seq_type == 'v']) / (n_v * 5))
    )


def execute_sift(image_path):
    return sift.compute(image_path)


def read_function(seq_name, im_idx):
    image_path = f"{dataset_path}/{seq_name}/{im_idx}.ppm"
    return execute_sift(image_path)


def evaluate():
    method = "own_sift"
    errors[method] = benchmark_features(read_function)
    summary(errors[method][-1])


if __name__ == "__main__":
    evaluate()
