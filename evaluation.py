import numpy as np
import torch
import kornia
from IPython.display import clear_output
from brown_phototour_revisited.benchmarking import *

from descriptor import compute_patch
from keypoints import compute_keypoint_hist, fit_parabola
from descriptor_matcher import evaluate


patch_size = 65
descs_out_dir = 'data/descriptors'
download_dataset_to = 'data/dataset'
results_dir = 'data/mAP'


def extract_sift(patch):
    bin_width = 10
    hist = compute_keypoint_hist(
        patch,
        patch.shape[0] // 2,
        patch.shape[1] // 2,
        bin_width
    )
    max_bin = np.argmax(hist)
    angle = fit_parabola(hist, max_bin, bin_width)
    desc = compute_patch((255*patch).astype(np.uint8), angle=angle, blur=True)
    return desc


def apply_first_evaluation():
    model = kornia.feature.SIFTDescriptor(patch_size, rootsift=False).eval()
    results_dict = {}
    results_dict['Kornia SIFT'] = full_evaluation(
        model,
        'Kornia SIFT',
        path_to_save_dataset=download_dataset_to,
        path_to_save_descriptors=descs_out_dir,
        path_to_save_mAP=results_dir,
        patch_size=patch_size,
        device=torch.device('cuda:0'),
        distance='euclidean',
        backend='pytorch-cuda'
    )

    results_dict['SIFT from scratch'] = full_evaluation(
        extract_sift,
        'SIFT from scratch',
        path_to_save_dataset=download_dataset_to,
        path_to_save_descriptors=descs_out_dir,
        path_to_save_mAP=results_dir,
        patch_size=patch_size,
        device=torch.device('cuda:0'),
        distance='euclidean',
        backend='pytorch-cuda'
    )
    clear_output()
    print_results_table(results_dict)


def apply_second_evaluation():
    evaluate()


if __name__ == "__main__":
    apply_first_evaluation()
