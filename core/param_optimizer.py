"""
Now that the model structure is correct, the final step for
fine-tuning and feature extracting is to create an optimizer
that only updates the desired parameters.
Recall that after loading the pretrained model,
but before reshaping, if feature_extract=True we manually set all of
the parameter’s .requires_grad attributes to False. Then the reinitialized
layer’s parameters have .requires_grad=True by default. So now we know that
all parameters that have .requires_grad=True should be optimized.
Next, we make a list of such parameters and input this list to the
SGD algorithm constructor.

To verify this, check out the printed parameters to learn.
When fine-tuning, this list should be long and include all of the model
parameters. However, when feature extracting this list should be short
and only include the weights and biases of the reshaped layers.
"""
from __future__ import print_function

from typing import Optional

from torch import optim

from core.imgnet_models.base_model import ImgnetModel


def optimize_params(
        model: ImgnetModel,
        feature_extract: bool
):
    model_ft = model.model_ft
    print("Params to learn:")
    if feature_extract:
        params_to_update = [
            param
            for name, param in model_ft.named_parameters()
            if param.requires_grad
        ]
    else:
        params_to_update = model_ft.parameters()

    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            print('\t', name)

    optimized_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    return optimized_ft
