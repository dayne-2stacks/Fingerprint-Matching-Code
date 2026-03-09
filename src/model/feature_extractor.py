import torch
import torch.nn as nn
from torchvision import models


class _TorchvisionResNetBase(nn.Module):
    _MODEL_NAME = None
    _WEIGHTS_ENUM_NAME = None
    _WEIGHTS_MEMBER = None
    _FINAL_LAYERS_DEFAULT = False

    def __init__(self, final_layers=None):
        super().__init__()
        if final_layers is None:
            final_layers = self._FINAL_LAYERS_DEFAULT
        self.node_layers, self.edge_layers, self.final_layers = self.get_backbone()
        if not final_layers:
            self.final_layers = None
        self.backbone_params = list(self.parameters())

    def forward(self, *inputs):
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def _build_torchvision_backbone(cls):
        backbone_ctor = getattr(models, cls._MODEL_NAME)
        try:
            weights_enum = getattr(models, cls._WEIGHTS_ENUM_NAME)
            weights = getattr(weights_enum, cls._WEIGHTS_MEMBER)
            return backbone_ctor(weights=weights)
        except AttributeError:
            return backbone_ctor(pretrained=True)

    @classmethod
    def get_backbone(cls):
        backbone = cls._build_torchvision_backbone()
        node_layers = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu,
            backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3
        )
        edge_layers = nn.Sequential(backbone.layer4)
        final_layers = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)))
        return node_layers, edge_layers, final_layers


class ResNet34_base(_TorchvisionResNetBase):
    """
    Base class that exposes ResNet-18 feature maps exactly like VGG16_base does:
      • node_layers – stride-16 feature map  (C=256, H/16, W/16)
      • edge_layers – stride-32 feature map  (C=512, H/32, W/32)
      • final_layers – optional 1×1 global feature (AdaptiveMaxPool)
    """
    _MODEL_NAME = "resnet34"
    _WEIGHTS_ENUM_NAME = "ResNet34_Weights"
    _WEIGHTS_MEMBER = "IMAGENET1K_12"
    _FINAL_LAYERS_DEFAULT = True

    def __init__(self, final_layers: bool = True):
        super().__init__(final_layers=final_layers)


class ResNet50_base(_TorchvisionResNetBase):
    """
    Base class that exposes ResNet-18 feature maps exactly like VGG16_base does:
      • node_layers – stride-16 feature map  (C=256, H/16, W/16)
      • edge_layers – stride-32 feature map  (C=512, H/32, W/32)
      • final_layers – optional 1×1 global feature (AdaptiveMaxPool)
    """
    _MODEL_NAME = "resnet50"
    _WEIGHTS_ENUM_NAME = "ResNet50_Weights"
    _WEIGHTS_MEMBER = "IMAGENET1K_V2"
    _FINAL_LAYERS_DEFAULT = True

    def __init__(self, final_layers: bool = True):
        super().__init__(final_layers=final_layers)




class ResNet18_base(_TorchvisionResNetBase):
    """
    Base class that exposes ResNet-18 feature maps exactly like VGG16_base does:
      • node_layers – stride-16 feature map  (C=256, H/16, W/16)
      • edge_layers – stride-32 feature map  (C=512, H/32, W/32)
      • final_layers – optional 1×1 global feature (AdaptiveMaxPool)
    """
    _MODEL_NAME = "resnet18"
    _WEIGHTS_ENUM_NAME = "ResNet18_Weights"
    _WEIGHTS_MEMBER = "IMAGENET1K_V1"
    _FINAL_LAYERS_DEFAULT = False

    def __init__(self, final_layers: bool = False):
        super().__init__(final_layers=final_layers)


# Convenience subclasses to mirror the VGG variants --------------------------

class ResNet18_final(ResNet18_base):
    """ResNet-18 **with** the final global-pool layer."""
    def __init__(self):
        super().__init__(final_layers=True)


class ResNet18(ResNet18_base):
    """ResNet-18 **without** the final global-pool layer."""
    def __init__(self):
        super().__init__(final_layers=False)

class VGG16_base(nn.Module):
    r"""
    The base class of VGG16. It downloads the pretrained weight by torchvision API, and maintain the layers needed for
    deep graph matching models.
    """
    def __init__(self, batch_norm=True, final_layers=False):
        super(VGG16_base, self).__init__()
        self.node_layers, self.edge_layers, self.final_layers = self.get_backbone(batch_norm)
        if not final_layers: self.final_layers = None
        self.backbone_params = list(self.parameters())

    def forward(self, *input):
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def get_backbone(batch_norm):
        """
        Get pretrained VGG16 models for feature extraction.

        :return: feature sequence
        """
        if batch_norm:
            model = models.vgg16_bn(pretrained=True)
        else:
            model = models.vgg16(pretrained=True)

        conv_layers = nn.Sequential(*list(model.features.children()))

        conv_list = node_list = edge_list = []

        # get the output of relu4_2(node features) and relu5_1(edge features)
        cnt_m, cnt_r = 1, 0
        for layer, module in enumerate(conv_layers):
            if isinstance(module, nn.Conv2d):
                cnt_r += 1
            if isinstance(module, nn.MaxPool2d):
                cnt_r = 0
                cnt_m += 1
            conv_list += [module]

            #if cnt_m == 4 and cnt_r == 2 and isinstance(module, nn.ReLU):
            if cnt_m == 4 and cnt_r == 3 and isinstance(module, nn.Conv2d):
                node_list = conv_list
                conv_list = []
            #elif cnt_m == 5 and cnt_r == 1 and isinstance(module, nn.ReLU):
            elif cnt_m == 5 and cnt_r == 2 and isinstance(module, nn.Conv2d):
                edge_list = conv_list
                conv_list = []

        assert len(node_list) > 0 and len(edge_list) > 0

        # Set the layers as a nn.Sequential module
        node_layers = nn.Sequential(*node_list)
        edge_layers = nn.Sequential(*edge_list)
        final_layers = nn.Sequential(*conv_list, nn.AdaptiveMaxPool2d((1, 1), return_indices=False)) # this final layer follows Rolink et al. ECCV20

        return node_layers, edge_layers, final_layers
    

class VGG16_bn_final(VGG16_base):
    r"""
    VGG16 with batch normalization and final layers.
    """
    def __init__(self):
        super(VGG16_bn_final, self).__init__(True, True)


class VGG16_bn(VGG16_base):
    r"""
    VGG16 with batch normalization, without final layers.
    """
    def __init__(self):
        super(VGG16_bn, self).__init__(True, False)


class VGG16_final(VGG16_base):
    r"""
    VGG16 without batch normalization, with final layers.
    """
    def __init__(self):
        super(VGG16_final, self).__init__(False, True)


class VGG16(VGG16_base):
    r"""
    VGG16 without batch normalization or final layers.
    """
    def __init__(self):
        super(VGG16, self).__init__(False, False)


class NoBackbone(nn.Module):
    r"""
    A model with no CNN backbone for non-image data.
    """
    def __init__(self, *args, **kwargs):
        super(NoBackbone, self).__init__()
        self.node_layers, self.edge_layers = None, None

    def forward(self, *input):
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device
