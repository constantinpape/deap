import torch

from torchvision.transforms import v2
from torchvision.transforms.functional import resize, crop, InterpolationMode
from torchvision import transforms
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation




def gaussian(d, sigma):
    return torch.exp(-0.5 * (d / sigma) ** 2) / (sigma * torch.sqrt(torch.tensor(2.0 * torch.pi)))



def pad_to_square(img):
    assert img.ndim==3
    assert img.shape[0] in {1,3}
    img_size = torch.tensor(img.shape[1:])

    pad = img_size.max() - img_size
    pad = torch.cat([pad // 2, pad - pad // 2])[[0,2,1,3]]
    return torch.nn.functional.pad(img, pad.tolist(), mode='constant', value=0)



def pad_to_square_pil(img):

    from PIL import ImageOps
    max_side = max(img.size)
    return ImageOps.pad(
        img,
        size=(max_side, max_side),
        color=0,
        centering=(0.5, 0.5), 
    )    

def prepare_labels(mask, sample_id, label_types, min_label_value=None, instance_count=None, device='cpu'):

    if label_types is None:
        label_types = ['sem', 'ids', 'offset', 'offset_geo', 'centers_gauss', 'centers_gauss_edt', 'centers_on', 'sizes', 'colorization', 'boundaries', 'foreground']

    label_types = set(label_types)

    n_frames, _, _ = mask.shape

    m_ = mask
    all_labels = set(m_.flatten().tolist())

    if min_label_value is not None:
        # this is primarily meant for VIPSeg where stuff segments are also encoded over ids
        inst_labels = [l for l in all_labels if l > min_label_value]
    else:
        inst_labels = list(all_labels)


    output = dict()

    output['raw_ids'] = mask

    if 'sem' in label_types or 'sem_boundaries' in label_types:
        m_sem = m_ * (m_ < 100).byte()
        m_sem += (m_ > 200)*(m_ // 100).byte()
        output['sem'] = m_sem

    if 'sem_boundaries' in label_types:
        # Note, these are based on semantic boundaries, not instances
        import skimage
        # assert self.n_frames == 1, 'not tested for more frames.'
        output['sem_boundaries'] = find_boundaries(output['sem'])


    if 'ids' in label_types:

        if instance_count is not None:
            shift = instance_count[sample_id]
        else:
            # this should only happen during validation
            shift = 0

        ids = torch.zeros(*mask.shape, dtype=torch.long)

        for i, l in enumerate(inst_labels):
            ids[m_==l] = shift + (i+1)


        output['ids'] = ids

    grid = torch.stack(
        torch.meshgrid(torch.linspace(0,1,mask.shape[1]), torch.linspace(0,1,mask.shape[2]), indexing='ij')
    ).permute(1,2,0)

    if 'foreground' in label_types:
        fg = output['ids'][0] > 0
        output['foreground'] = fg[None]


    if 'boundaries' in label_types:

        assert 'ids' in output
        h, w = mask.shape[1:]
        bd = find_boundaries(output['ids'][0], connectivity=9)
        bd = dilation(bd, torch.ones(h // 100, w // 100).numpy())
        output['boundaries'] = torch.from_numpy(bd[None]).float()

    if 'offset' in label_types:

        from scipy.ndimage import distance_transform_edt

        output['offsets_on'] = torch.zeros(*mask.shape, 2)
        output['offsets_on2'] = torch.zeros(*mask.shape, 2)

        assert 'ids' in output
        assert 'foreground' in output
        
        # make sure the max is not at the edge of the image
        bd = torch.nn.functional.pad(torch.from_numpy(bd), (1, 1, 1, 1), value=1).numpy()
        edt_boundaries = distance_transform_edt(~bd)
        edt_boundaries = edt_boundaries[1:-1, 1:-1]

        output['centers_edt'] = torch.from_numpy(edt_boundaries * fg.float().numpy()[None]).float()

        offsets = []
        assert output['ids'].shape[0] == 1
        for i, l in enumerate(inst_labels):
            this_mask = m_ == l
            
            a = output['centers_edt']*(this_mask)
            means = torch.tensor([a.argmax() // w, a.argmax() % w]) * torch.tensor([1/a.shape[1], 1/a.shape[2]])[None]
            means = torch.nan_to_num(means)    
            offs = (grid - means[0, None, None, :])[None]
            offsets += [offs]
            output['offsets_on'][this_mask] = offs[this_mask]
            output['offsets_on2'][this_mask] = offs[this_mask] / offs[this_mask].pow(2).sum(1).sqrt()[...,None]

            # normalize centers
            c = output['centers_edt'][this_mask]
            output['centers_edt'][this_mask] = c / (c.max() +0.01)

        
        output['centers_edt'] = output['centers_edt'].float()
        output['offsets_on'] = output['offsets_on'].permute(3,0,1,2).to(device)
        output['offsets_on2'] = output['offsets_on2'].permute(3,0,1,2).to(device)

    if 'centers_on' in label_types:
        assert 'offset' in label_types
        output['centers_on'] = torch.zeros(*mask.shape)
        for i, (l, off) in enumerate(zip(inst_labels, offsets)):
            # this_mask = m_ == l      
            output['centers_on'][0, off[0].pow(2).sum(2).sqrt() < 0.01] = 1

    if 'centers_gauss_edt' in label_types:
        assert 'offset' in label_types
        output['centers_gauss_edt'] = torch.zeros(*mask.shape)
        #output['centers_gauss'] = torch.zeros(600, 600, 2)
        for i, (l, off) in enumerate(zip(inst_labels, offsets)):

            this_mask = m_ == l
            s = this_mask.float().mean().sqrt()

            s = 0.12*s
            
            # Objects as Points implementation
            d = off[:,:,:,0].pow(2) + off[:,:,:,1].pow(2)
            output['centers_gauss_edt'] += torch.exp(-d/(2*s.pow(2)))

            # output['centers_gauss'] += s*gaussian(off[0].pow(2).sum(2).sqrt(), 0.1*s)

        output['centers_gauss_edt'] = output['centers_gauss_edt'].clip(0, 1)
        

    if 'centers_gauss' in label_types:
        output['centers_gauss'] = torch.zeros(*mask.shape)
        output['centers_size'] = torch.zeros(*mask.shape, 2)

        grid_ = grid.mul(mask.shape[1]).round().div(mask.shape[1])

        for i, l in enumerate(inst_labels):

            this_mask = m_ == l
            means = torch.stack([grid[m_[k]==l].mean(0) for k in range(n_frames)])
            # print(i, means)
            means = torch.nan_to_num(means)
            dists = (grid[:,:,:,None] - means.T).permute(3,0,1,2)

            coords = torch.argwhere(this_mask[0])
            
            sy = (coords[:,0].max() - coords[:,0].min()) / this_mask.shape[1]
            sx = (coords[:,1].max() - coords[:,1].min()) / this_mask.shape[2]
            s = this_mask.float().mean().sqrt()

            s = 0.12*s * (224.0 / this_mask.shape[2])

            # s = 0.02*s

            means = means.mul(mask.shape[1]).round().div(mask.shape[1])
            

            for ii, m in enumerate(means):
                d = (grid_ - m[None, None, :]).pow(2).sum(2)

                output['centers_gauss'] += torch.exp(-d/(2*s.pow(2)))
                output['centers_size'][0, d.sqrt()<0.01,:] = torch.tensor([sx, sy]).float()

        output['centers_gauss'] = output['centers_gauss'].clip(0, 1)
        output['centers_size'] =  output['centers_size'].permute(3,0,1,2)


    if 'sizes' in label_types:
        output['sizes'] = torch.zeros(*mask.shape)
        # output['sizes'] = output['sizes'] * (~bd).astype('float32')
        for i, l in enumerate(inst_labels):
            this_mask = m_ == l
            s = this_mask.sum() / output['sizes'].numel()
            output['sizes'][this_mask] = s            

    if 'offset_geo' in label_types:
        output['offsets_geo'] = torch.zeros(*mask.shape, 2)
        output['centers_geo'] = torch.zeros(*mask.shape)
        for i, l in enumerate(inst_labels):
            means = torch.stack([grid[m_[k]==l].mean(0) for k in range(n_frames)])
            # print(i, means)
            means = torch.nan_to_num(means)
            dists = (grid[:,:,:,None] - means.T).permute(3,0,1,2)
            output['offsets_geo'][m_==l] = dists[m_==l]
            
            #s = (m_ == l).sum() / sizes.numel()
            
            # es = int(max(1, s*300))
            # m2 = torch.from_numpy(binary_erosion((m_ == l)[0], footprint=torch.ones(es, es).bool().numpy()))
            # sizes[m_==l] = s

            # sizes[m_==l] = s
            for ii, m in enumerate(means):
                m = (grid - m[None, None, :]).pow(2).sum(2).sqrt()
                output['centers_geo'][ii, m < 0.02] += 1 - m[m < 0.02] / 0.02
                # sizes[ii, m < 0.02] = s
                #sizes[m_==l] = 1

        output['offsets_geo'] = output['offsets_geo'].permute(3,0,1,2).to(device)
        #output['sizes'] = sizes
        

    if 'colorization' in label_types:
        colors = torch.randn(len(inst_labels), 3)
        colors = torch.nn.functional.normalize(colors)        
        color_mask = torch.zeros(*m_.shape, 3)

        for i, l in enumerate(inst_labels):
            
            color_mask[(m_==l)] = colors[i]
        
        output['colorization'] = color_mask
        
        # dm[m==l] *= dist[m==l].abs().sum(1, keepdim=True)

    return output



class ColorJitterOnImage:
    def __init__(self):
        self.transform = v2.ColorJitter(hue=(-0.02,0.02), brightness=(0.8, 1.2))

    def __call__(self, sample):
        if torch.rand((1,)).item() > 0.5:
            sample['image'] = self.transform(sample['image'])
        return sample
    


def joint_image_mask_random_crop(image, mask, target_size=224, upscale_fac=1.5, mask_interpolation=InterpolationMode.NEAREST):

    assert image.ndim == 3 and mask.ndim == 2

    if isinstance(upscale_fac, (tuple, list)):
        upscale_fac = torch.distributions.Uniform(upscale_fac[0], upscale_fac[1]).sample((1,)).item()

    image = resize(image, int(target_size*upscale_fac))
    m = resize(mask[None], int(target_size*upscale_fac), interpolation=mask_interpolation)[0]
    m_sum = m.sum()
    for _ in range(20):
        i, j, h, w = transforms.RandomCrop.get_params(image, (target_size, target_size))
        m2 = crop(m, i ,j,h, w)
        if m2.sum() > 0.3*m_sum:
            break
    m = m2
    image = crop(image, i ,j,h, w)
    return image, m


def joint_image_mask_augment(image, mask, img_size, p_gray=0.3, p_hflip=0.5, p_col=0.5, brightness=(0.3, 1.5), contrast=(0.3, 1.5), saturation=(0.3, 1.5), upscale_fac=(1,1.5), mask_interpolation=InterpolationMode.NEAREST):
    
    if torch.rand(1).item() < p_gray:
        image = v2.functional.rgb_to_grayscale(image, num_output_channels=3)

    if torch.rand(1).item() < p_col:
        image = v2.functional.adjust_brightness(image, torch.distributions.Uniform(*brightness).sample().item())
        image = v2.functional.adjust_contrast(image, torch.distributions.Uniform(*contrast).sample().item())
        image = v2.functional.adjust_saturation(image, torch.distributions.Uniform(*saturation).sample().item())

    image, mask = joint_image_mask_random_crop(image, mask, target_size=img_size, upscale_fac=upscale_fac, mask_interpolation=mask_interpolation)

    if torch.rand(1).item() < p_hflip:
        image = image.flip(-1)
        mask = mask.flip(-1)

    return image, mask