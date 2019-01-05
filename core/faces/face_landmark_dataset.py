import os
from typing import Callable, Optional
import pandas as pd
from skimage import io

from torch.utils.data import Dataset

from core.faces.transform_funcs import IdFunc
from core.faces.typing import Sample


class FaceLandmarksDataset(Dataset):
    def __init__(
            self,
            csv_file: str,
            root_dir: str,
            transform: Optional[Callable] = None
    ):
        """
        :param csv_file:
           Path to the csv file with annotations
        :param root_dir:
           Directory with all the images
        :param transform:
           A callable object
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        if transform is None:
            self.transform = IdFunc()
        else:
            self.transform = transform

    def __getitem__(self, index) -> Sample:
        img_name = self.landmarks_frame.iloc[index, 0]
        img_file_name = os.path.join(self.root_dir, img_name)
        img = io.imread(img_file_name)
        landmarks = self.landmarks_frame.iloc[index, 1:].values
        landmarks = landmarks.astype(pd.np.float).reshape(-1, 2)
        return self.transform(Sample(image=img, landmarks=landmarks))

    def __len__(self):
        return len(self.landmarks_frame)
