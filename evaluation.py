import cv2
import numpy as np
import torch
import kornia
from IPython.display import clear_output
from brown_phototour_revisited.benchmarking import *

from descriptor import compute_patch


def extract_sift(patch):
    desc = compute_patch((255*patch).astype(np.uint8), angle=0)
    return desc


patch_size = 65

model = kornia.feature.SIFTDescriptor(patch_size, rootsift=False).eval()

descs_out_dir = 'data/descriptors'
download_dataset_to = 'data/dataset'
results_dir = 'data/mAP'

results_dict = {}
# results_dict['Kornia SIFT'] = full_evaluation(
#     model,
#     'Kornia SIFT',
#     path_to_save_dataset=download_dataset_to,
#     path_to_save_descriptors=descs_out_dir,
#     path_to_save_mAP=results_dir,
#     patch_size=patch_size,
#     device=torch.device('cuda:0'),
#     distance='euclidean',
#     backend='pytorch-cuda'
# )



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
