import dataclasses
import numpy as np
from collections import Callable
from typing import Tuple, Union

from skimage import transform

from core.faces.typing import Sample
import torch


class IdFunc(Callable):
    """
     The IdFunc does nothing, just return input sample
    """

    def __call__(self, sample: Sample):
        return sample


class Rescale(Callable):
    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        self.output_size = output_size

    def __call__(self, sample: Sample):
        image, landmarks = dataclasses.astuple(sample)
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * (h / w), self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * (w / h)
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        new_img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axis are axis 1 and axis 0 respectively
        new_landmarks = landmarks * [new_w / w, new_h / h]

        return Sample(
            image=new_img,
            landmarks=new_landmarks
        )


class RandomCrop(Callable):
    """
      Crop randomly the image in a sample
    """

    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample: Sample) -> Sample:
        image, landmarks = dataclasses.astuple(sample)
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h, left: left + new_w]

        landmarks = landmarks - [left, top]

        return Sample(
            image=image,
            landmarks=landmarks
        )


class ToTensor(Callable):
    def __call__(self, sample: Sample):
        image, landmarks = dataclasses.astuple(sample)

        # swap color axis because
        # numpy image: H * W * C
        # torch image: C * H * W
        image = image.transpose((2, 0, 1))
        return Sample(
            image=image,
            landmarks=landmarks
        )
