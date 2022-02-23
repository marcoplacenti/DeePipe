from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.make_dataset import ImageDataset


if __name__ == "__main__":
    dataset = ImageDataset(
                    data_dir="./data/image-classification/", 
                    transform=transforms.Resize((48,48)))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
