import os
import sys
sys.path.append('../../code_release')

import torch

from deap.models.dense_attentive_probing import SelfAttReadouts
from deap.datasets.pascal_simple import PascalVOC12Segmentation

from types import SimpleNamespace


def train(cfg, device='cuda'):

    model = cfg['model'].init()
    dataset= cfg['dataset'].init()
    task = cfg['task'].init()

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.bs, shuffle=True, num_workers=cfg.n_workers, pin_memory=True
    )

    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)

    model.to(device)

    iteration, continue_training = 0, True

    while continue_training:

        print('iteration', iteration)
        for sample in loader:

            model.train()

            opt.zero_grad()
            
            outputs = model(sample)
            loss, _ = task.loss_function(outputs, sample)

            loss.backward() 
            opt.step()

            if iteration >= cfg.n_iterations:
                continue_training = False
                break

            iteration += 1   

    torch.save({k:p for k,p in model.state_dict().items() if p.requires_grad}, cfg['name'] + '-weights.pth')


def evaluate(cfg, weights, device='cuda'):

    model = cfg['model'].init()
    model.load_state_dict(torch.load(weights), strict=False)
    model.to(device)

    dataset= cfg['dataset_test'].init()
    task = cfg['task'].init()
    
    out = task.evaluate(model, dataset)
    print(out)
    return out


if __name__ == '__main__':

    args = SimpleNamespace(bs=16, lr=0.001, wd=0.001, n_workers=4)
        
    model = SelfAttReadouts('vit_b-mae', 
        base_size=28, # query grid resolution
        up=(4,2), #upscaling in the CNN
        inp_img_size=224, # input image size
        dim=32, # internal dimension, controls number of parameters
        decoder='CA-A3-sl-only-mask', # masked attention type
        outputs=[('seg', 21)] # output definition,
    )

    os.environ['PASCAL_VOC_PATH'] = '/scratch/datasets/pascal_voc2012'
    dataset = PascalVOC12Segmentation('train')

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.bs, shuffle=True, num_workers=args.n_workers, pin_memory=True
    )

    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    device = 'cuda'
    model.to(device)

    for sample in loader:

        model.train()

        opt.zero_grad()
        
        outputs = model(sample)
        loss = torch.nn.functional.cross_entropy(
            outputs['seg'][:,:,0].permute(1,0,2,3).flatten(1).cpu().T,
            sample['sem'].flatten().long(),
            ignore_index=255
        )

        loss.backward()