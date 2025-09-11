
import torch
from matplotlib import pyplot as plt
from torchvision.ops import focal_loss

from deap.tasks.base import OnlyLossEvaluation

### Mixin definitions

class BCELoss(object):

    loss_fn = staticmethod(torch.nn.functional.binary_cross_entropy_with_logits)

    def loss_function(self, outputs, labels, iteration=None): 

        out_scaled = torch.nn.functional.interpolate(
            outputs[self.key_name], 
            labels[self.key_name].shape[1:4],
            mode='trilinear'
        )

        out_scaled = out_scaled[:, 0]  # there is only a single map
        device = out_scaled.device
        
        loss = self.loss_fn(
            out_scaled.flatten(), 
            labels[self.key_name].flatten().to(device).float()
        )
        return loss, dict()
    

class FocalLoss(object):


    def loss_function(self, outputs, labels, iteration=None): 

        out_scaled = torch.nn.functional.interpolate(
            outputs[self.key_name], 
            labels[self.key_name].shape[1:4],
            mode='trilinear'
        )

        out_scaled = out_scaled[:, 0]  # there is only a single map
        device = out_scaled.device
        
        loss = focal_loss.sigmoid_focal_loss(
            out_scaled.flatten(), 
            labels[self.key_name].flatten().to(device).float(),
            reduction='mean'
        )

        return loss, dict()


class MSELoss(BCELoss):

    loss_only_valid = False

    def loss_function(self, outputs, labels, iteration=None, reduce=True): 

        if outputs[self.key_name].shape[2:5] != labels[self.key_name].shape[1:4]:
            import warnings
            warnings.warn(f'rescale from {outputs[self.key_name].shape[2:5]} to {labels[self.key_name].shape[1:4]}')

            assert outputs[self.key_name].shape[2] == 1

            out_scaled = torch.nn.functional.interpolate(
                outputs[self.key_name], 
                labels[self.key_name].shape[1:4],
                mode='trilinear',
            )
        else:
            out_scaled = outputs[self.key_name]

        out_scaled = out_scaled[:, 0]  # there is only a single map
        device = out_scaled.device
        
        if self.loss_only_valid:
            loss = torch.nn.functional.mse_loss(
                out_scaled.flatten(1), 
                labels[self.key_name].flatten(1).to(device).float(),
                reduction='none'
            )

            valid = (labels[self.key_name].flatten(1).to(device) != 0).float()
            loss = (loss * valid).sum(1) / (0.00001 + valid.sum(1))
        else:
            # print(out_scaled.shape, labels[self.key_name].shape)
            loss = torch.nn.functional.mse_loss(
                out_scaled.flatten(1), 
                labels[self.key_name].flatten(1).to(device).float(),
            )

        if reduce:
            # average over batch
            loss = loss.mean()

        return loss, dict()



class SimplePlots(object):

    def plot(self, sample, out=None):

        n_cols = 2 if out is None else 3
        fig, ax = plt.subplots(2,n_cols, figsize=(2*n_cols, 3))
        
        bs = sample['image'].shape[0]
        for i in range(min(bs, 2)):

            # image
            ax[i, 0].imshow(sample['image'][i,:,0].permute(1,2,0))

            # offsets in RGB
            ax[i, 1].imshow(sample[self.key_name][i,0], interpolation='nearest')

            if out is not None:
                ax[i, 2].imshow(out[self.key_name][i,0,0].cpu().detach(), interpolation='nearest')

            [a.axis('off') for a in ax.flatten()]

        fig.tight_layout()
        plt.close(fig)
        return fig       


### Task definitions

class BoundaryPredictionTask(BCELoss):

    def __init__(self, threshold=0.5, key_name='boundaries'):
        self.label_types = ['boundaries']
        self.key_name = key_name
        self.threshold = threshold

    def evaluate(self, model, dataset_val, bs=8, n_workers=1):

        from torchmetrics.classification import JaccardIndex
        
        model.eval()
        metric_jaccard = JaccardIndex("binary", threshold=self.threshold)
        losses_val = []

        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=bs, shuffle=False, num_workers=n_workers)

        with torch.no_grad():
                
            for sample_val in loader_val:

                outputs = model(sample_val)

                # scale model output to label size
                preds = torch.nn.functional.interpolate(
                    outputs[self.key_name].cpu(), 
                    sample_val[self.key_name].shape[-3:],
                    mode='trilinear'
                )
                
                metric_jaccard.update(preds.squeeze(1).flatten(), sample_val[self.key_name].flatten())
                losses_val += [self.loss_function(outputs, sample_val)[0]]

        score_jaccard = metric_jaccard.compute()
        
        out = dict(loss = torch.tensor(losses_val).mean().item())

        if not torch.isnan(score_jaccard):
            out.update(iou=score_jaccard.item())

        return out


    def plot(self, sample, out=None):

        n_cols = 2 if out is None else 3
        fig, ax = plt.subplots(2,n_cols, figsize=(2*n_cols, 3))
        
        bs = sample['image'].shape[0]
        for i in range(min(bs, 2)):

            # image
            ax[i, 0].imshow(sample['image'][i,:,0].permute(1,2,0))

            # offsets in RGB
            ax[i, 1].imshow(sample[self.key_name][i,0], interpolation='nearest')

            if out is not None:
                ax[i, 2].imshow(out[self.key_name][i,:,0].detach().cpu().argmax(0),
                    cmap=plt.cm.rainbow, interpolation='nearest', vmin=0, vmax=120)

            [a.axis('off') for a in ax.flatten()]

        fig.tight_layout()
        plt.close(fig)
        return fig


class DepthPredictionTask(MSELoss, SimplePlots):

    loss_only_valid = True

    def __init__(self):
        self.label_types = ['depth']
        self.key_name = 'depth'
     
    def evaluate(self, model, dataset_val, bs=8, n_workers=1):

        model.eval()
        losses_val, rmse, diffs, valid = [], [], [], []

        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=bs, shuffle=False, num_workers=n_workers, drop_last=False)

        with torch.no_grad():
                
            for sample_val in loader_val:

                outputs = model(sample_val)

                out = torch.nn.functional.interpolate(outputs['depth'][:,:,0], sample_val['depth'].shape[2:])
                diff = sample_val['depth'] - out.cpu().detach()
                diffs += [diff]
                valid += [(sample_val['depth'] <= 10) & (sample_val['depth'] > 1e-3)]

                loss_out = self.loss_function(outputs, sample_val, reduce=False)
                losses_val += [loss_out[0].mean()]

                # rmse += loss_out[0].tolist()

        valid_diffs = torch.cat(diffs)[torch.cat(valid)]
        # for RMSE calculation, we follow 
        # https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/blob/main/depth/core/evaluation/metrics.py#L46
        out = dict(loss = torch.tensor(losses_val).mean().item(), rmse=valid_diffs.pow(2).mean().sqrt().item())
        return out
    


class DistanceTransformTask(MSELoss, OnlyLossEvaluation, SimplePlots):

    def __init__(self):
        # self.label_types = ['depth']
        self.key_name = 'centers_edt'


class CenterNetTaskFocal(FocalLoss, OnlyLossEvaluation, SimplePlots):

    def __init__(self):
        # self.label_types = ['depth']
        self.key_name = 'centers_gauss'

