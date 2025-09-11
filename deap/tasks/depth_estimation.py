import torch


def sigloss(input, target, valid_mask=False, lam=0.15, min_depth=0, max_depth=None, eps=0.001):
    """ this is based on the sigloss implementation from DINOv2 """

    if valid_mask:
        valid_mask = target > 0
        if max_depth is not None:
            valid_mask = torch.logical_and(target > min_depth, target <= max_depth)
        input = input[valid_mask]
        target = target[valid_mask]

    g = torch.log(input + eps) - torch.log(target + eps)
    Dg = torch.var(g) + lam * torch.pow(torch.mean(g), 2)
    return torch.sqrt(Dg)



class DepthEvaluate(object):
    def evaluate(self, model, dataset_val, bs=8, n_workers=1):

        import tqdm
        
        model.eval()
        losses_val, diffs, valid = [], [], []

        loader_val =torch.utils.data.DataLoader(dataset_val, batch_size=bs, shuffle=False, num_workers=n_workers, drop_last=False)

        with torch.no_grad():
                
            for sample_val in  tqdm.tqdm(loader_val):

                outputs = model(sample_val)

                # import ipdb; ipdb.set_trace()

                out = torch.nn.functional.interpolate(
                    outputs['depth'][:,:,0], 
                    sample_val['depth'].shape[2:],
                    mode='bilinear'
                )
                diff = sample_val['depth'] - out.cpu().detach()
                diffs += [diff]
                # Note, it matters a lot if this is a < or <=
                valid += [(sample_val['depth'] < 10) & (sample_val['depth'] > 1e-3)]

                loss_out = self.loss_function(outputs, sample_val, reduce=False)
                losses_val += [loss_out[0].mean()]

        valid = torch.cat(valid)[:,0]
        diffs = torch.cat(diffs)[:,0]

        # use eigen crop
        a = torch.zeros(valid.shape[0], 480, 640).bool()
        a[:, 45:471, 41:601] = True
        valid = torch.logical_and(a, valid)

        valid_diffs = diffs[valid]
        # for RMSE calculation, we follow 
        # https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/blob/main/depth/core/evaluation/metrics.py#L46
        out = dict(loss = torch.tensor(losses_val).mean().item(), rmse=valid_diffs.pow(2).mean().sqrt().item())
        return out





class DepthPredictionTaskSigloss(DepthEvaluate):

    def __init__(self, lam=0.15):
        self.label_types = ['depth']
        self.key_name = 'depth'
        self.lam = lam

    def loss_function(self, outputs, labels, iteration=None, reduce=None): 

        # reduce is ignored, but we use valid mask

        import sys
        
        assert outputs[self.key_name].shape[2] == 1

        out_scaled = torch.nn.functional.interpolate(
            outputs[self.key_name][:,:,0], 
            labels[self.key_name].shape[2:4],
            mode='bilinear'
        )

        device = out_scaled.device
        
        loss = sigloss(
            out_scaled.relu().flatten(), 
            labels[self.key_name].flatten().to(device).float(),
            valid_mask=True,
            min_depth=1e-3,
            max_depth=10,
            lam=self.lam  # used to be 0.15
        )
        return loss, dict()
    