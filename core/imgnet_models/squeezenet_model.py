from torch import nn
from torchvision import models

from core.imgnet_models.base_model import ImgnetModel
from core.types import ModelName
from core.utils import set_parameter_requires_grad


class SqueezenetModel(ImgnetModel):
    def __init__(self):
        self._model_ft = None

    def init(
            self,
            device,
            num_classes: int,
            feature_extract: bool,
            use_pretrained: bool = True
    ):
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extracting=feature_extract)
        model_ft.classifier[1] = nn.Conv2d(
            512,
            num_classes,
            kernel_size=(1, 1),
            stride=(1, 1)
        )
        model_ft.num_classes = num_classes
        self._model_ft = model_ft.to(device)

    @property
    def model_ft(self):
        return self._model_ft

    @property
    def input_size(self) -> int:
        return 224

    @property
    def model_name(self) -> ModelName:
        return ModelName.Squeezenet
