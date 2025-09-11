import os
import torchvision
from torchvision import transforms
from deap.datasets.base import BaseDataset


class PascalVOC12Segmentation(BaseDataset):

    def __init__(self, split, chunks=None, label_types=None, img_size=224, max_samples=None):
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        target_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.PILToTensor()
        ])

        self.split = split

        subset_names = {
            'train': 'train',
            'trainval': 'trainval',
            'val': 'val',
        }

        self.dataset = torchvision.datasets.VOCSegmentation(
            os.environ['PASCAL_VOC_PATH'], 
            transform=transform, 
            image_set=subset_names[split],
            target_transform=target_transform
        )

        if max_samples is not None:
            self.dataset.images = self.dataset.images[:max_samples]

    def sample(self, bs=4, shuffle=False):
        from torch.utils.data import DataLoader
        return next(iter(DataLoader(self, batch_size=bs, shuffle=shuffle)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return dict(
            image=sample[0].unsqueeze(1),
            sem=sample[1],
            id=self.split + '-' + str(idx)
        )
