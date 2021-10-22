# ------------------------------------------------------------------------------
# Modified based on torchvision.models.alexnet.
# ------------------------------------------------------------------------------

import tourch
import torch.nn as nn
from torchvision import models
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.alexnet import model_urls
import copy

__all__ = ["AlexNet", "alexnet"]

class AlexNet(models.AlexNet):
    """MoibleNets without fully connected layer"""

    def __init__(self, *args, **kwargs):
        super(AlexNet, self).__init__(*args, **kwargs)
        self._out_features = self.classifier[6].in_features

    def forward(self, x):
        f = self.features(x)
        f = self.avgpool(f)
        f = torch.flatten(f, 1)
        predictions = self.classifier(f)
        return predictions, f

    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self._out_features

    def copy_head(self) -> nn.Module:
        """Copy the origin fully connected layer"""
        return copy.deepcopy(self.classifier)

def alexnet(num_classes: int, pretrained: bool = False, progress: bool = True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    The required minimum input size of the model is 63x63.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "alexnet" 
    model = AlexNet(**kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        # remove keys from pretrained dict that doesn't appear in model dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)
        model.classifier[6] = nn.Linear(model.out_features,num_classes)
    return model


