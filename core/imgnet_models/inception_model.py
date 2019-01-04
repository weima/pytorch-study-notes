from torch import nn
from torchvision import models

from core.imgnet_models.base_model import ImgnetModel
from core.types import ModelName
from core.utils import set_parameter_requires_grad


class InceptionModel(ImgnetModel):
    def __init__(self):
        self._model_ft = None

    def init(
            self,
            device,
            num_classes: int,
            feature_extract: bool,
            use_pretrained: bool = True
    ):
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extracting=feature_extract)
        # Handle the auxiliary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)

        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

        self._model_ft = model_ft.to(device)

    @property
    def model_ft(self):
        return self._model_ft

    @property
    def input_size(self) -> int:
        return 299

    @property
    def model_name(self) -> ModelName:
        return ModelName.Inception
