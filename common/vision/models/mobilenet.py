# ------------------------------------------------------------------------------
# Modified based on torchvision.models.mobilenet.
# ------------------------------------------------------------------------------

import torch.nn as nn
from torchvision import models
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf, model_urls
import copy

__all__ = ["MobileNetV3", "mobilenet_v3_large", "mobilenet_v3_small"]


class MobileNetV3(models.MobileNetV3):
    """MoibleNets without fully connected layer"""

    def __init__(self, *args, **kwargs):
        super(MobileNetV3, self).__init__(dropout = 0.2, *args, **kwargs)
        cls_in_features = self.classifier[0].in_features
        self._out_features = self.classifier[3].in_features

        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(cls_in_features, self._out_features),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True)
        )    

    def _forward_impl(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs):
    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        # remove keys from pretrained dict that doesn't appear in model dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)
    return model        

def mobilenet_v3_large(pretrained: bool = False, progress: bool = True, **kwargs):
    """
    Constructs a large MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_large"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, **kwargs)
    return _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs)


def mobilenet_v3_small(pretrained: bool = False, progress: bool = True, **kwargs):
    """
    Constructs a small MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_small"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, **kwargs)
    return _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs)