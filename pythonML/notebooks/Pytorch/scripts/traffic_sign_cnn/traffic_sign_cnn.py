from torch.utils.data import Dataset
from PIL import Image
class TrafficSignDataset(Dataset):
    def __init__(self, data_array, labels, transform=None):
        """
        Args:
            data_array (numpy array): with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_array = data_array
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        image = Image.fromarray(self.data_array[idx])
        if self.transform:
            image = self.transform(image)

        return image, int(self.labels[idx])