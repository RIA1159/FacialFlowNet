import os
from typing import Dict,Any, List
import torch
import yaml
import torch.nn as nn
from pytorchcv.model_provider import get_model
import albumentations as A
import cv2
import numpy as np
def load_yaml(x: str) -> Dict[str, Any]:
    with open(x) as fd:
        config = yaml.load(fd, yaml.FullLoader)
        return config

class Encoder(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True, config_name: str = "backbone.yaml") -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = self._load_model()
        self.stages = self._get_stages()
        config = load_yaml(os.path.join(os.path.dirname(__file__), config_name))
        self.encoder_channels = config[model_name].get("block_size", None)
        self.final_block_channels = config[model_name].get("final_block_channels", None)
    def _load_model(self) -> Any:
        model = get_model(self.model_name, pretrained=self.pretrained).features

        model = nn.Sequential(
            model.init_block, 
            model.stage1,
            model.stage2,
            model.stage3
            )


        return model

    def _get_stages(self) -> List[Any]:
        stages = [
            nn.Sequential(self.model[0], self.model[1]),
            self.model[2],
            self.model[3],
        ]
        return stages

    def forward(self, x: Any) -> List[Any]:
        encoder_maps = []
        for stage in self.stages:
            x = stage(x)
            encoder_maps.append(x)
        return encoder_maps


class StagedEncoder(Encoder):
    def __init__(self, model_name: str, pretrained: bool = True, config_name: str = "backbone.yaml") -> None:
        super().__init__(model_name=model_name, pretrained=pretrained, config_name=config_name)

    def _get_stages(self) -> List[Any]:
        stages = [self.model[0], self.model[1], self.model[2]]
        return stages


encoder_mapping = {
    "resnet50": StagedEncoder,
    "mobilenet_w1": Encoder
}


def get_encoder(encoder_name: str = "resnet50", pretrained: bool = True, config_name: str = "backbone.yaml") -> Encoder:
    encoder = encoder_mapping[encoder_name](encoder_name, pretrained, config_name)
    for name, param in encoder.named_parameters():
        param.requires_grad = False
    return encoder





def dad_3dhead_encoder(encoder,image):

    x = image
    for stage in encoder.stages:
        x = stage(x)

    return x


