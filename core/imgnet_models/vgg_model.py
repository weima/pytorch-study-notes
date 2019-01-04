from torch import nn
from torchvision import models

from core.imgnet_models.base_model import ImgnetModel
from core.types import ModelName
from core.utils import set_parameter_requires_grad


class VggModel(ImgnetModel):
    def __init__(self):
        self._model_ft = None

    def init(
            self,
            device,
            num_classes: int,
            feature_extract: bool,
            use_pretrained: bool = True
    ):
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extracting=feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        self._model_ft = model_ft.to(device)

    @property
    def model_ft(self):
        return self._model_ft

    @property
    def input_size(self) -> int:
        return 224

    @property
    def model_name(self) -> ModelName:
        return ModelName.Vgg
