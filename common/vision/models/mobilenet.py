# ------------------------------------------------------------------------------
# Modified based on torchvision.models.mobilenet.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf, model_urls
import copy

__all__ = ["MobileNetV3", "mobilenet_v3_large", "mobilenet_v3_small"]


class MobileNetV3(models.MobileNetV3):
    """MoibleNets without fully connected layer"""

    def __init__(self, *args, **kwargs):
        super(MobileNetV3, self).__init__(*args, **kwargs)
        self._out_features = self.classifier[3].in_features

    def _forward_impl(self, x):
        f = self.features(x)
        f = self.avgpool(f)
        f = torch.flatten(f, 1)
        predictions = self.classifier(f)

        return predictions, f

    def forward(self, x):
        return self._forward_impl(x)

    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self._out_features

    def copy_head(self) -> nn.Module:
        """Copy the origin fully connected layer"""
        return copy.deepcopy(self.classifier)

def _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs):
    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        # remove keys from pretrained dict that doesn't appear in model dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)
    return model        

def mobilenet_v3_large(num_classes: int, pretrained: bool = False, progress: bool = True,  **kwargs):
    """
    Constructs a large MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_large"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, **kwargs)
    model = _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs)
    model.classifier[3] = nn.Linear(model.out_features, num_classes)
    return model

def mobilenet_v3_small(num_classes: int, pretrained: bool = False, progress: bool = True, **kwargs):
    """
    Constructs a small MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_small"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, **kwargs)
    model = _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs)
    model.classifier[3] = nn.Linear(model.out_features, num_classes)
    return model