import os
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import imageio.v2 as imageio
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim
from skimage.segmentation import slic
from skimage.filters import threshold_otsu
from sklearn.metrics.pairwise import pairwise_distances

from model.Mask2Former import Mask2FormerFeatureExtractor
from model.MODEL import Mulit_layer_Encoder, GCN_Edge
from aux_func.preprocess import preprocess_img
from aux_func.graph_func import construct_affinity_matrix
from aux_func.chart_func import visualize_semantic_feature, visualize_segmentation
from aux_func.clustering import otsu
from aux_func.acc_ass import assess_accuracy


def load_checkpoint_for_evaluation(model, checkpoint):
    saved_state_dict = torch.load(checkpoint, map_location='cuda:0', weights_only=True)
    model.load_state_dict(saved_state_dict)
    model.cuda()
    model.eval()


def segment_lcs(img_t1, img_t2):
    img_seg = np.concatenate((img_t1, img_t2), axis=2).astype(np.float32)
    img_norm = np.zeros_like(img_seg, dtype=np.uint8)
    for c in range(img_norm.shape[2]):
        img_seg[:, :, c] = cv2.GaussianBlur(img_seg[:, :, c], (3, 3), 0)
        img_norm[:, :, c] = cv2.normalize(img_seg[:, :, c], None, 0, 255, cv2.NORM_MINMAX)
    img_norm = img_norm.astype(np.uint8)

    spx = cv2.ximgproc.createSuperpixelLSC(
        image=img_norm,
        region_size=args.seg_size,
        ratio=0.1,
    )
    spx.iterate(10)
    return spx.getLabels(), spx.getLabelContourMask()


def train_model():
    ### load dataset
    img_t1 = imageio.imread('data/T1.png')
    img_t2 = imageio.imread('data/T2.png')
    gt_changed = imageio.imread('data/GT.png')
    gt_unchanged = 255 - gt_changed

    height, width, channel_t1 = img_t1.shape
    print(f"Shape of img_t1: {img_t1.shape}")
    _, _, channel_t2 = img_t2.shape


    ### image segmentation
    objects, edge_mask = segment_lcs(img_t1, img_t2)  # LSC
    # objects = slic(img_t2, n_segments=1500, compactness=5, start_label=0)  # SLIC
    obj_num = objects.max() + 1
    # visualize_segmentation(img_t1, img_t2, objects)
    print(f"Number of objects: {obj_num}")


    ### image preprocess
    FeatureExtractor = Mask2FormerFeatureExtractor()
    m2f_feats_t1 = FeatureExtractor.extract_all_features(img_t1, upsample=False)
    m2f_feats_t2 = FeatureExtractor.extract_all_features(img_t2, upsample=False)
    m2f_feats_t1 = [torch.tensor(m2f_feats_t1["multi_scale_encoder"][0]).cuda().float().permute(2, 0, 1).unsqueeze(0),
                    torch.tensor(m2f_feats_t1["multi_scale_encoder"][1]).cuda().float().permute(2, 0, 1).unsqueeze(0),
                    torch.tensor(m2f_feats_t1["multi_scale_decoder"][1]).cuda().float().permute(2, 0, 1).unsqueeze(0)]
    m2f_feats_t2 = [torch.tensor(m2f_feats_t2["multi_scale_encoder"][0]).cuda().float().permute(2, 0, 1).unsqueeze(0),
                    torch.tensor(m2f_feats_t2["multi_scale_encoder"][1]).cuda().float().permute(2, 0, 1).unsqueeze(0),
                    torch.tensor(m2f_feats_t2["multi_scale_decoder"][1]).cuda().float().permute(2, 0, 1).unsqueeze(0)]
    img_t1 = preprocess_img(img_t1, d_type='sar', norm_type='stad')
    img_t2 = preprocess_img(img_t2, d_type='opt', norm_type='stad')
    img_t1 = torch.from_numpy(img_t1).cuda().float().permute(2, 0, 1).unsqueeze(0)
    img_t2 = torch.from_numpy(img_t2).cuda().float().permute(2, 0, 1).unsqueeze(0)

    MultiLayerEncoder = Mulit_layer_Encoder()
    MLE_path = './model_weight/pretrain_edge.pth'
    load_checkpoint_for_evaluation(MultiLayerEncoder, MLE_path)
    out1, feat1 = MultiLayerEncoder(img_t1, m2f_feats_t1)
    out2, feat2 = MultiLayerEncoder(img_t2, m2f_feats_t2)
    # visualize_semantic_feature(out1.detach().cpu().numpy())
    # visualize_semantic_feature(out2.detach().cpu().numpy())


    ### graph construction
    feat1 = feat1.cpu().detach().numpy()
    feat2 = feat2.cpu().detach().numpy()
    out1 = out1.cpu().detach().numpy()
    out2 = out2.cpu().detach().numpy()
    node_set_t1, node_set_t2 = [], []
    for idx in range(obj_num):
        obj_idx = objects == idx
        node_set_t1.append(feat1[obj_idx])
        node_set_t2.append(feat2[obj_idx])

    diff_set = []
    for _iter in range(obj_num):
        node_t1 = node_set_t1[_iter]
        node_t2 = node_set_t2[_iter]

        node_t1 = torch.from_numpy(node_t1).cuda().float()
        node_t2 = torch.from_numpy(node_t2).cuda().float()

        graph_t1 = torch.matmul(node_t1, node_t1.T)
        graph_t2 = torch.matmul(node_t2, node_t2.T)
        diff = torch.mean(torch.abs(node_t1 - node_t2))
        diff_set.append(diff.data.cpu().numpy())

    diff_map = np.zeros((height, width))
    for i in range(obj_num):
        diff_map[objects == i] = diff_set[i]
    diff_map = np.reshape(diff_map, (height * width, 1))
    threshold = otsu(diff_map)
    diff_map = np.reshape(diff_map, (height, width))
    bcm = np.zeros((height, width)).astype(np.uint8)
    bcm[diff_map > threshold] = 255
    bcm[diff_map <= threshold] = 0

    conf_mat, oa, f1, kappa_co = assess_accuracy(gt_changed, gt_unchanged, bcm)

    imageio.imsave('./result/Edge_' + str(time.time()) + '.png', bcm)
    diff_map = 255 * (diff_map - np.min(diff_map)) / (np.max(diff_map) - np.min(diff_map))
    imageio.imsave('./result/Edge_' + str(time.time()) + '_DI.png', diff_map.astype(np.uint8))

    print(conf_mat)
    print(f'OA: {oa}, F1: {f1}, Kappa: {kappa_co}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RS Change Detection')
    parser.add_argument('--seg_size', type=int, default=12, help='Approximate size of segments')
    parser.add_argument('--epochs', type=int, default=30, help='Epoch number of model training')
    parser.add_argument('--band_width_t1', type=float, default=0.5, help='The bandwidth of Gaussian kernel when calculating the adj_mat of t1')
    parser.add_argument('--band_width_t2', type=float, default=0.5, help='The bandwidth of Gaussian kernel when calculating the adj_mat of t2')

    args = parser.parse_args()
    train_model()
