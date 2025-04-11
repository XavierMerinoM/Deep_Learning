from PIL import Image
from torch.utils.data import Dataset

# Class to save a dataset of a single image 
class SingleImageDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return 1  # Only one image

    def __getitem__(self, idx):
        image = Image.open(self.image_path)
        if self.transform:
            image = self.transform(image)
        return image