import torch
import math
import time


def get_sample(dataset, bs=1, shuffle=False):
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=bs, shuffle=shuffle)
    return next(iter(loader))


def square_meshgrid(size, device='cpu'):
    return torch.dstack(
        torch.meshgrid(torch.linspace(0, 1, size, device=device), 
                       torch.linspace(0, 1, size, device=device), indexing='xy')
    )

def rect_meshgrid(sizes, device='cpu'):
    return torch.dstack(
        torch.meshgrid(torch.linspace(0, 1, sizes[0], device=device), 
                       torch.linspace(0, 1, sizes[1], device=device), indexing='xy')
    )


def random_base64_string(length):
    """ credits to GPT3.5 """

    import random
    import string
    import base64

    random_bytes = bytes(''.join(random.choices(string.ascii_letters + string.digits, k=length)), 'utf-8')
    return base64.b64encode(random_bytes).decode('utf-8')[:length]


class TimeBlock(object):

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print(f'enter block {self.name}...')
        self.t_start = time.time()

    def __exit__(self, type, value, traceback):
        print(f'block {self.name} took {time.time() - self.t_start:.1f}s')



def pos_enc_sincos(x, dim=10, flatten=False):
    """ 
    sinosoidal positional encoding 
    important: value range of x must be considered. This configuration works best in [-100, 100].
    """

    old_shape = x.shape
    x = x.flatten()
    # div_vec = 1 / (10000 ** (2 * torch.arange(dim).float() / dim))
    div_vec = torch.exp(torch.arange(0, dim, 2).to(x.device) * (-math.log(10000.0) / dim))

    pe2 = torch.cat([
        torch.sin(x[:,None] * div_vec[None, :]),
        torch.cos(x[:,None] * div_vec[None, :]),
    ], dim=1)

    pe2 = pe2.view(old_shape + (dim,))

    if flatten:
        pe2 = pe2.flatten(-2, -1)
        
    return pe2




def count_parameters(model, only_trainable=False):
    """ Count the number of parameters of a torch model. """
    import numpy as np
    return sum([np.prod(p.size()) for p in model.parameters()
                if (only_trainable and p.requires_grad) or not only_trainable])


def show_types(sample):

    if isinstance(sample, dict):
        x = list(sample.items())
    elif isinstance(sample, tuple):
        print(f'list of {len(sample)} elements')
        x = list(enumerate(sample[:20]))
    elif isinstance(sample, list):
        print(f'list of {len(sample)} elements')
        x = list(enumerate(sample[:20]))

    for k, v in x:
        if isinstance(v, torch.Tensor):
            print(f'{k:<12}{str(v.dtype):<20}{str(v.shape)}')
        else:
            if hasattr(v, '__len__'):
                print(f'{k:<12}{str(type(v).__name__):<20}length: {str(len(v))}')


def show(*imgs):
    from matplotlib import pyplot as plt

    def process_img(img):

        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()


        if img.ndim == 2:
            if isinstance(img, (torch.FloatTensor)):
                return dict(X=img)
            else:    
                return dict(X=img, cmap=plt.cm.tab20, vmin=0, interpolation='nearest')

        # quick hack for offsets
        if img.shape[0] == 2:
            img = torch.cat([torch.zeros(1, *img.shape[1:]), img]).permute(1,2,0)   

        assert img.ndim == 3

        if img.shape[0] == 3:
            img = img.permute(1,2,0)


        return dict(X=img)

    _ , ax = plt.subplots(1, len(imgs), figsize=(len(imgs)*4, 5))

    if len(imgs) == 1:
        ax.axis('off')
        ax.imshow(**process_img(imgs[0]))
    else:
        [a.axis('off') for a in ax.flatten()]
        for i, img in enumerate(imgs):
            ax[i].imshow(**process_img(img))