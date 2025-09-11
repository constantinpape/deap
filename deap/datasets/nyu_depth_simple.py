import torch
import os
import numpy as np

from deap.datasets.functions import prepare_labels, ColorJitterOnImage
# from deap import files


class NYUDepthV2(object):

    def __init__(self, split, img_size, aug=(), eigen_crop=False, max_samples=None, index_stride=None):
        
        # assert isinstance(img_size, int)
        assert isinstance(aug, (tuple, list))

        self.max_samples = max_samples
        self.eigen_crop = eigen_crop

        if split == 'train_all':
            split2, indices = 'train', range(0, 47000)
        elif split == 'train':
            split2, indices = 'train', range(0, 42000)                     
        elif split == 'val':
            split2, indices = 'train', range(42000, 47000)
        elif split == 'val2':
            split2, indices = 'train', range(42000, 47000, 10)                  
        elif split == 'test':
            split2, indices = 'validation', range(0, 654)

        self.split = split2

        if max_samples is not None:
            indices = indices[:max_samples]

        if index_stride is not None:
            indices = indices[::index_stride]

        self.sample_ids = list(indices)

        # files.from_s3('hf_sayakpaul_nyu_depth_v2', file_type='tar')
        
        from datasets import load_dataset, load_from_disk
        self.dataset = load_from_disk(os.path.join(os.environ['NYU_DEPTH_PATH'], 'sayakpaul_nyu_depth_v2'))
        self.dataset = self.dataset[split2]
        # self.dataset = load_dataset("sayakpaul/nyu_depth_v2", split=split2)

        from torchvision.transforms import v2

        if len(aug) > 0:
            self.transforms = [
                v2.ToImage()
            ]

            if 'hf' in aug:
                self.transforms += [v2.RandomHorizontalFlip(p=0.5)]                

            self.transforms += [
                v2.ToDtype(torch.float32, scale=True),
            ]
        else:
            self.transforms = [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]

        self.transforms = v2.Compose(self.transforms)

        print(self.transforms)

    def __len__(self):
        if self.max_samples is not None:
            return min(len(self.sample_ids), self.max_samples)
        else:
            return len(self.sample_ids)

    def sample(self, bs=4, shuffle=False):
        from torch.utils.data import DataLoader
        return next(iter(DataLoader(self, batch_size=bs, shuffle=shuffle)))

    def __getitem__(self, i):
        j = self.sample_ids[i]
        sample = self.dataset[j]

        # image, depth = sample['image'], sample['depth_map']
        data_tf = self.transforms(sample)  

        out = dict()
        out['image'] = data_tf['image'][:,None]
        out['depth'] = data_tf['depth_map']

        # map all invalid values to zero
        out['depth'][out['depth'] > 10] = 0
        out['depth'][out['depth'] < 1e-3] = 0

        if self.eigen_crop:
            assert out['image'].shape[2:] == (480, 640)
            assert out['depth'].shape[1:3] == (480, 640)
            out['image'] = out['image'][:, :, 45:471, 41:601]
            out['depth'] = out['depth'][:, 45:471, 41:601]

        out['id'] = {'train':'0', 'validation': '1'}[self.split] + str(j)
        return out
