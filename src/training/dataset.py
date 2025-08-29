import torch
from torch.utils.data import Dataset
from PIL import Image


class FengShuiDataset(Dataset):
    def __init__(self, image_paths, labels, preprocessor):
        self.image_paths = image_paths
        self.labels = labels
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        preprocessed = self.preprocessor.preprocess(image_path, augment=True)

        return {
            'pixel_values': preprocessed['tensor'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }