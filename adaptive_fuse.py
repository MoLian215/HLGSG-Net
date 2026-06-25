import imageio.v2 as imageio
import numpy as np
import cv2
from aux_func.acc_ass import assess_accuracy
from aux_func.clustering import otsu


def fuse_DI_1():
    LocalGCN_DI = imageio.imread('./result/Ver_DI.png').astype(np.float32)
    NLocalGCN_DI = imageio.imread('./result/Edge_DI.png').astype(np.float32)
    height, width = LocalGCN_DI.shape
    alpha = np.var(LocalGCN_DI.reshape(-1))
    beta = np.var(NLocalGCN_DI.reshape(-1))
    fuse_DI = (alpha * LocalGCN_DI + beta * NLocalGCN_DI) / (alpha + beta)
    fuse_DI = np.reshape(fuse_DI, (height * width, 1))
    threshold = otsu(fuse_DI)
    fuse_DI = np.reshape(fuse_DI, (height, width))
    bcm = np.zeros((height, width)).astype(np.uint8)
    bcm[fuse_DI > threshold] = 255
    bcm[fuse_DI <= threshold] = 0
    imageio.imsave('./result/Adaptive_Fuse.png', bcm)

    fuse_DI = 255 * ((fuse_DI - np.min(fuse_DI)) / (np.max(fuse_DI) - np.min(fuse_DI)))
    imageio.imsave('./result/Adaptive_Fuse_DI.png', fuse_DI.astype(np.uint8))

    ground_truth_changed = imageio.imread('data/GT.png')
    ground_truth_unchanged = 255 - ground_truth_changed
    conf_mat, oa, f1, kappa_co = assess_accuracy(ground_truth_changed, ground_truth_unchanged, bcm)
    print(conf_mat)
    print(f'OA: {oa}, F1: {f1}, Kappa: {kappa_co}')


def fuse_DI_2():
    LocalGCN_DI = imageio.imread('./result/Ver_DI.png').astype(np.float32)
    NLocalGCN_DI = imageio.imread('./result/Edge_DI.png').astype(np.float32)
    height, width = LocalGCN_DI.shape

    alpha = np.var(LocalGCN_DI.reshape(-1))
    beta = np.var(NLocalGCN_DI.reshape(-1))
    fuse_DI = (alpha * LocalGCN_DI + beta * NLocalGCN_DI) / (alpha + beta)

    ver_norm = 255 * ((LocalGCN_DI - LocalGCN_DI.min()) / (LocalGCN_DI.max() - LocalGCN_DI.min() + 1e-8))
    edge_norm = 255 * ((NLocalGCN_DI - NLocalGCN_DI.min()) / (NLocalGCN_DI.max() - NLocalGCN_DI.min() + 1e-8))
    ver_norm = ver_norm.astype(np.uint8)
    edge_norm = edge_norm.astype(np.uint8)

    _, Ver_bin = cv2.threshold(
        ver_norm,
        0,
        1,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    _, Edge_bin = cv2.threshold(
        edge_norm,
        0,
        1,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 交集(高置信区域)
    intersection = (
            (Ver_bin == 1) &
            (Edge_bin == 1)
    ).astype(np.uint8)
    # 并集(候选变化区域)
    union = (
            (Ver_bin == 1) |
            (Edge_bin == 1)
    ).astype(np.uint8)
    # 差集(不确定区域)
    diff_region = union - intersection

    inverse_intersection = 1 - intersection
    dist = cv2.distanceTransform(
        inverse_intersection,
        cv2.DIST_L2,
        5
    )
    weight_map = np.ones((height, width), dtype=np.float32)

    high_gain = 0.5
    low_gain = 1.8
    tau = 15.0

    weight_map[intersection == 1] = 1.0 + high_gain

    decay = np.exp(-dist / tau)
    weight_map[diff_region == 1] = 1.0 + low_gain * decay[diff_region == 1]
    fuse_DI = fuse_DI * weight_map

    weight_vis = 255 * (weight_map - np.min(weight_map)) / (np.max(weight_map) - np.min(weight_map))
    imageio.imsave('./result/weight_map.png', weight_vis.astype(np.uint8))

    fuse_DI = fuse_DI.reshape(height * width, 1)
    threshold = otsu(fuse_DI)
    fuse_DI = fuse_DI.reshape(height, width)
    bcm = np.zeros((height, width), dtype=np.uint8)
    bcm[fuse_DI > threshold] = 255
    bcm[fuse_DI <= threshold] = 0

    imageio.imsave('./result/Adaptive_Fuse.png', bcm)
    fuse_DI = 255 * ((fuse_DI - np.min(fuse_DI)) / (np.max(fuse_DI) - np.min(fuse_DI)))
    imageio.imsave('./result/Adaptive_Fuse_DI.png', fuse_DI.astype(np.uint8))

    ground_truth_changed = imageio.imread('data/GT.png')
    ground_truth_unchanged = 255 - ground_truth_changed
    conf_mat, oa, f1, kappa_co = assess_accuracy(ground_truth_changed, ground_truth_unchanged, bcm)
    print(conf_mat)
    print(f'OA: {oa}, F1: {f1}, Kappa: {kappa_co}')


if __name__ == '__main__':
    fuse_DI_2()
