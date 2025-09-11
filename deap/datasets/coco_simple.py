from torchvision.transforms import v2
from torchvision.datasets import CocoDetection

import torchvision
import torch
import os

from deap.datasets.functions import prepare_labels
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image


class COCO(object):

    def __init__(self, split, img_size, inp_img_size=None, aug=False, label_types=None,
                 max_samples=None, instance_count=None, special_filter=None, cache=False):
        
        assert isinstance(img_size, int)

        self.label_types = label_types
        self.img_size = (img_size, img_size)
        self.aug = aug
        self.max_samples = max_samples
        self.inp_img_size = inp_img_size

        if isinstance(instance_count, str):
            instance_count = torch.load(instance_count)

        self.instance_count = instance_count

        if aug:
            self.transforms = v2.Compose([
                v2.ToImage(),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomResizedCrop(img_size, antialias=True),
                v2.ToDtype(torch.float32, scale=True),
            ])
        else:
            self.transforms = v2.Compose([
                v2.ToImage(),
                v2.Resize(img_size, antialias=True),
                v2.CenterCrop(img_size),
                v2.ToDtype(torch.float32, scale=True),
                # v2.CenterCrop()
            ])            

        self.split = split
        subset_name, self.sample_ids = {
            'train': ('train2017', range(0, 100000)), 
            'val': ('train2017', range(100000, 118287)),
            'test': ('val2017', range(0, 5000)),
        }[split]

        self.sample_ids = list(self.sample_ids)

        
        import os
        

        self.coco = CocoDetection(
            os.path.join(os.environ['COCO_PATH'], subset_name), 
            annFile=os.path.join(os.environ['COCO_PATH'], f'annotations/instances_{subset_name}.json'),
            transforms=self.transforms
        )
        self.coco = torchvision.datasets.wrap_dataset_for_transforms_v2(self.coco, target_keys=["boxes", "labels", "masks"])

        if special_filter == 'at_least_three_big_objects':
            coco = self.coco
            self.sample_ids = [j for j in self.sample_ids 
                               if sum([coco.coco.anns[i]['area'] > 15000 for i in coco.coco.getAnnIds([coco.ids[j]])])> 2]
            
            self.sample_ids = self.sample_ids[:5000]
            print('reduce to ', len(self.sample_ids))

        self.cache = dict() if cache else None



    def __len__(self):
        if self.max_samples is not None:
            return min(len(self.sample_ids), self.max_samples)
        else:
            return len(self.sample_ids)

    def sample(self, bs=4, shuffle=False):
        from torch.utils.data import DataLoader
        return next(iter(DataLoader(self, batch_size=bs, shuffle=shuffle)))

    def __getitem__(self, i):

        if self.cache is not None and self.sample_ids[i] in self.cache:
            return self.cache[self.sample_ids[i]]
        else:
            s = self.coco[self.sample_ids[i]]

            sample_id = self.split + '-' + str(i)

        
            # TODO: quick hack, can be speeded-up
            m = torch.cat([torch.zeros(1, *self.img_size)] +  ([s[1]['masks']] if 'masks' in s[1] else []))
            # print(m.argmax(0)[None].max())

            out = prepare_labels(m.argmax(0)[None].int(), i, self.label_types, min_label_value=0,
                                instance_count=self.instance_count)
            out['image'] = s[0][:,None]  # add dim for compatibility
            if self.inp_img_size is not None:
                out['image'] = torch.nn.functional.interpolate(out['image'], self.inp_img_size, mode='bilinear')
            out['id'] = sample_id

            if self.cache is not None:
                self.cache[self.sample_ids[i]] = out

            return out


    

class COCOStuff(object):

    def __init__(self, split, aug=False, img_size=224, max_samples=None):
        
        subset_name, self.sample_ids = {
            'train': ('train2017', range(0, 100000)), 
            'val': ('train2017', range(100000, 118287)),
            'test': ('val2017', range(0, 5000)),
        }[split]

        self.stuff_path = os.path.join(os.environ['COCO_STUFF_PATH'], 'trainval2017', subset_name)
        self.samples = os.listdir(self.stuff_path)
        self.samples = [f[:12] for f in self.samples]
        
        self.coco_path = os.path.join(os.environ['COCO_PATH'],  subset_name)    
        self.max_samples = max_samples  
        if aug:
            self.transforms = v2.Compose([
                v2.ToImage(),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomResizedCrop(img_size, antialias=True),
                v2.ToDtype(torch.float32, scale=True),
            ])
        else:
            self.transforms = v2.Compose([
                v2.ToImage(),
                v2.Resize(img_size, antialias=False),
                v2.CenterCrop(img_size),
                v2.ToDtype(torch.float32, scale=True),
            ])   

    def __len__(self):
        if self.max_samples is not None:
            return min(len(self.sample_ids), self.max_samples)
        else:
            return len(self.sample_ids)

    def __getitem__(self, idx):
        
        sample = self.samples[idx]
        seg = pil_to_tensor(Image.open(os.path.join(self.stuff_path, sample + '.png')).convert('RGB'))[0]
        img = pil_to_tensor(Image.open(os.path.join(self.coco_path, sample + '.jpg')).convert('RGB'))

        from torchvision import tv_tensors

        out = dict(image=tv_tensors.Image(img), label=tv_tensors.Mask(seg))
        out = self.transforms(out)

        out['image'] = out['image'][:,None]
        out['sem'] = out['label'][None]
        out['id'] = sample
        # del out['label']

        return out