import torch

from core.types import TorchDevice

device = torch.device(TorchDevice.Cuda if torch.cuda.is_available() else TorchDevice.Cpu)


def set_parameter_requires_grad(model, feature_extracting):
    """
    This helper function sets the .requires_grad attribute of
    the parameters in the model to False when we are feature extracting.
    By default when loading a pre-trained model, all parameters have
    .requires_grad=True, which is fine if training from scratch or
    fine-tuning.
    However, if we are feature extracting and only want to compute gradients for
    the newly initialized layer, we want all parameters to NOT require gradients.

    :param model:
    :param feature_extracting:
            Performing feature extracting?
    :return:
    """
    needs_grad = not feature_extracting
    for param in model.parameters():
        param.requires_grad = param.requires_grad and needs_grad
