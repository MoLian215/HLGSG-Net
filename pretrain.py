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

from model.Mask2Former import Mask2FormerFeatureExtractor
from model.MODEL import Mulit_layer_Encoder
from aux_func.loss_func import loss_func
from aux_func.preprocess import preprocess_img
from aux_func.chart_func import visualize_semantic_feature


def pretrain():
    ### load dataset
    img_t1 = imageio.imread('data/T1.png')
    img_t2 = imageio.imread('data/T2.png')

    height, width, channel_t1 = img_t1.shape
    print(f"Shape of img_t1: {img_t1.shape}")
    _, _, channel_t2 = img_t2.shape


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
    img_t1_r = preprocess_img(img_t1, d_type='sar', norm_type='stad')
    img_t2_r = preprocess_img(img_t2, d_type='opt', norm_type='stad')
    img_t1_r = torch.from_numpy(img_t1_r).cuda().float()
    img_t2_r = torch.from_numpy(img_t2_r).cuda().float()
    # visualize_semantic_feature(img_t1)
    # visualize_semantic_feature(img_t2)


    ### pretrain
    MultiLayerEncoder = Mulit_layer_Encoder()
    optimizer = optim.Adam(MultiLayerEncoder.parameters(), lr=1e-4, weight_decay=1e-6)
    MultiLayerEncoder.cuda().train()

    img_t1 = img_t1_r.permute(2, 0, 1).unsqueeze(0)
    img_t2 = img_t2_r.permute(2, 0, 1).unsqueeze(0)

    for _epoch in range(args.epochs):
        optimizer.zero_grad()
        out1, feat1 = MultiLayerEncoder(img_t1, m2f_feats_t1)
        out2, feat2 = MultiLayerEncoder(img_t2, m2f_feats_t2)

        loss_img1, loss_branch1 = loss_func(out1, feat1, img_t1_r, branch=args.branch)
        loss_img2, loss_branch2 = loss_func(out2, feat2, img_t2_r, branch=args.branch)

        loss_img = loss_img1 + loss_img2
        loss_branch = loss_branch1 + loss_branch2

        total_loss = loss_img + loss_branch * 3
        total_loss.backward()
        optimizer.step()

        print(f'Epoch: {_epoch + 1}: Total_loss: {total_loss.item():.6f}, '
              f'Loss_img: {loss_img.item():.6f}, Loss_branch: {loss_branch.item():.6f}')

    torch.save(MultiLayerEncoder.state_dict(), './model_weight/pretrain_' + args.branch + '_' + str(time.time()) + '.pth')

    MultiLayerEncoder.eval()
    out1, feat1 = MultiLayerEncoder(img_t1, m2f_feats_t1)
    out2, feat2 = MultiLayerEncoder(img_t2, m2f_feats_t2)
    visualize_semantic_feature(out1.detach().cpu().numpy())
    visualize_semantic_feature(out2.detach().cpu().numpy())
    visualize_semantic_feature(feat1.detach().cpu().numpy())
    visualize_semantic_feature(feat2.detach().cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RS Change Detection')
    parser.add_argument('--epochs', type=int, default=20, help='Epoch number of pretraining')
    parser.add_argument('--branch', type=str, default="ver", help='Branch of multi-layer encoder to pretrain')

    args = parser.parse_args()
    pretrain()
