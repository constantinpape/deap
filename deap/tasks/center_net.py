
import torch
import os
from deap.tasks.base import OnlyLossEvaluation
from torchvision.ops import focal_loss


class CenterNetTask(OnlyLossEvaluation):

    def __init__(self, mode='focal'):
        self.mode = mode
        self.key_name = 'centers_gauss'

    def evaluate(self, model, dataset_val, bs=8, n_workers=1, min_score=0.01):

        import scipy.ndimage as ndi
        import numpy as np
        from torchmetrics.detection import MeanAveragePrecision
        from torchvision.ops import nms

        model.eval()
        ap = MeanAveragePrecision(iou_type="bbox")
        losses_val = []

        n_workers = min(os.cpu_count(), n_workers)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=bs, shuffle=False, num_workers=n_workers)

        with torch.no_grad():
                
            for sample_val in loader_val:

                pred = model(sample_val)
                
                img_size = pred['centers_gauss'].shape[3:]
                assert img_size[0] == img_size[1]

                for i in range(len(pred['centers_gauss'])):
                    pred_centers = pred['centers_gauss'][i,0,0].cpu().detach().sigmoid()
                    pred_sizes = pred['centers_size'][i,:,0].cpu().detach()

                    peaks = pred_centers.numpy() == ndi.maximum_filter(pred_centers.numpy(), size=3, mode='reflect')
                    peaks = peaks & (pred_centers.numpy() > min_score)
                    sizes = pred_sizes[:, peaks].T*img_size[0]
                    scores = pred_centers[peaks]
                    peaks = torch.from_numpy(np.argwhere(peaks)[:, ::-1].copy())

                    combined = list(zip(peaks, sizes, scores.tolist()))
                    combined = sorted(combined, key=lambda x: x[2], reverse=True)

                    if len(combined) > 0:
                        #boxes = torch.stack([torch.cat([p - 0.5*size, p + 0.5*size])  
                        #        for p, size, score in combined]).float()
                        boxes = torch.cat([peaks - 0.5* sizes, peaks + 0.5* sizes], dim=1)
                        boxes = boxes.clamp(0, img_size[0])
                        # scores = torch.tensor([s for _,_,s in combined])
                        valid_indices = nms(boxes, scores, 0.5)[:100]
                        boxes = boxes[valid_indices]
                        scores = scores[valid_indices]
                    else:
                        boxes, scores = torch.tensor([]), torch.tensor([])

                    sample_id = int(sample_val['id'][i].split('-')[1])
                    gt = dataset_val.coco[dataset_val.sample_ids[sample_id]]

                    gt_boxes = gt[1]['boxes'].data if 'boxes' in gt[1] else torch.tensor([])

                    ap.update(
                        [dict(boxes=boxes, scores=scores, labels=torch.tensor([0]*len(scores)))], 
                        [dict(boxes=gt_boxes, labels=torch.tensor([0]*len(gt_boxes)))]
                    )

                # metric_jaccard.update(preds.squeeze(1).flatten(), sample_val[self.key_name].flatten())
                losses_val += [self.loss_function(pred, sample_val)[0]]

        ap_scores = ap.compute()
        out = dict(loss = torch.tensor(losses_val).mean().item(), 
                   ap=ap_scores['map'].item(), 
                   ap_s=ap_scores['map_small'].item(),
                   ap_m=ap_scores['map_medium'].item(),
                   ap_lg=ap_scores['map_large'].item())

        return out    


    def loss_function(self, outputs, labels, iteration=None): 

        # out_scaled = torch.nn.functional.interpolate(
        #     outputs[self.key_name], 
        #     labels[self.key_name].shape[1:4],
        #     mode='trilinear'
        # )
        out_scaled = outputs[self.key_name]

        out_scaled = out_scaled[:, 0]  # there is only a single map
        device = out_scaled.device

        # old
        if self.mode == 'focal':
            loss = focal_loss.sigmoid_focal_loss(
                out_scaled.flatten(), 
                labels[self.key_name].flatten().to(device).float(),
                reduction='mean'
            )
        elif self.mode == 'mod_focal':
            alpha = 2
            beta = 4

            out_scaled = out_scaled.sigmoid()
            heatmap_gt = labels[self.key_name].to(device).float()
            pos_mask = (heatmap_gt == 1)
            neg_mask = ~pos_mask
            
            pos_loss = torch.pow(1 - out_scaled, alpha) * torch.log(out_scaled) * pos_mask.float()
            neg_loss = torch.pow(1 - heatmap_gt, beta) * torch.pow(out_scaled, alpha) * torch.log(1 - out_scaled) * neg_mask.float()

            n_objects = pos_mask.flatten(1).sum(1)

            # print(-pos_loss.flatten(1).sum(1))

            loss = (-pos_loss.flatten(1).sum(1) - neg_loss.flatten(1).sum(1)) / (1+n_objects[:, None, None, None])
            loss = loss.mean() # mean over samples
        elif self.mode == 'center_net':
            loss = _neg_loss(out_scaled.sigmoid(), labels[self.key_name].to(device))
        else:
            raise ValueError('invalid mode')

        labels_size = labels['centers_size'].to(device)
        valid = labels_size > 0
        loss_l1 = torch.nn.functional.l1_loss(
            labels_size, 
            outputs['centers_size'], 
            reduction='none'
        )
        
        loss_l1 = loss_l1[valid].mean()

        loss_complete = loss + loss_l1

        return loss_complete, dict(size=float(loss_l1), center=float(loss))




def _neg_loss(pred, gt):

    # copied from CenterNet codebase

  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss