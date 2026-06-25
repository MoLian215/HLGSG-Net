import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, Mask2FormerForUniversalSegmentation
import numpy as np


class Mask2FormerFeatureExtractor:
    _model_cache = {}

    def __init__(self, model_name="facebook/mask2former-swin-small-ade-semantic", device="cuda"):
        self.model_name = model_name
        self.device = device

        if model_name not in self._model_cache:
            processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True, do_resize=False)
            model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)

            model.to(self.device)
            model.eval()
            self._model_cache[model_name] = (model, processor)
        self.model, self.processor = self._model_cache[model_name]

    def _preprocess(self, img):
        if isinstance(img, np.ndarray):
            img_np = img
        elif hasattr(img, "convert"):
            img_np = np.array(img.convert("RGB"))
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        inputs = self.processor(images=img_np, return_tensors="pt")
        return inputs.to(self.device), img_np.shape[:2]

    def extract_feature(self, img, feature_type="pixel_last", upsample=False):
        """
        feature_type options:
            - "pixel_last": 最终融合特征
            - "multi_scale": 多尺度特征列表
        """
        inputs, (H, W) = self._preprocess(img)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

            # for i, feat in enumerate(outputs.encoder_hidden_states):
            #     print(i, feat.shape)

        if feature_type == "pixel_last":
            feat = outputs.pixel_decoder_last_hidden_state  # [B, C, H/4, W/4]
            feat = feat[0]

            if upsample:
                feat = torch.nn.functional.interpolate(
                    feat.unsqueeze(0),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False
                )[0]

            feat_np = feat.permute(1, 2, 0).cpu().numpy()
            return feat_np  # (H/4, W/4, 256)

        elif feature_type == "multi_scale_decoder":
            multi_feats = outputs.pixel_decoder_hidden_states  # list

            results = []
            for f in multi_feats:
                f = f[0]
                if upsample:
                    f = torch.nn.functional.interpolate(
                        f.unsqueeze(0),
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False
                    )[0]

                results.append(f.permute(1, 2, 0).cpu().numpy())

            return results  # ( , , 256)

        elif feature_type == "multi_scale_encoder":
            multi_feats = outputs.encoder_hidden_states  # list

            results = []
            for f in multi_feats:
                f = f[0]
                if upsample:
                    f = torch.nn.functional.interpolate(
                        f.unsqueeze(0),
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False
                    )[0]

                results.append(f.permute(1, 2, 0).cpu().numpy())

            return results  # ( , , 96/192/384/768)

        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

    def extract_all_features(self, img, upsample=False):
        """
        :param img: (H, W, 3)
        :return: (h, w, C) & list
        """
        features = {}
        feature_level = ["pixel_last", "multi_scale_decoder", "multi_scale_encoder"]
        for f in feature_level:
            features[f] = self.extract_feature(img, feature_type=f, upsample=upsample)

        return features
