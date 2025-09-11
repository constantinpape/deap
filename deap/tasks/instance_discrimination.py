import torch
import warnings
from torch import nn


def rand_index(pred, sample, res=(112, 112)):

    from sklearn.metrics import adjusted_rand_score, rand_score
    from sklearn.cluster import KMeans

    device = pred['instances'].device
    bs = pred['instances'].shape[0]

    bg = sample['ids'] == 0
    bg = nn.functional.interpolate(bg.byte(), res, mode='bilinear')[:,0]

    scores = []
    for i in range(bs):

        a = pred['instances'][i,:,0]
        a = nn.functional.interpolate(a[None], res, mode='bilinear')[0]

        n_objects = len(set(sample['ids'][i].flatten().tolist())) - 1

        l = torch.zeros(*res, dtype=torch.int)

        if n_objects > 0:
            # kmeans = KMeans(n_objects, n_init=5)
            # c = kmeans.fit_predict(a[:, bg[i] != 1].T.cpu())
            # l[bg[i] != 1] = (torch.from_numpy(c).int()+1)

            kmeans = KMeans(n_objects, n_init=15)
            c = kmeans.fit_predict(a.flatten(1).T.cpu()) + 1
            l = torch.from_numpy(c).view(*res) 
            valid = bg[i] != 1
            # l[~valid] = 0

        gt = nn.functional.interpolate(sample['ids'].byte(), res)[i,0]
        scores += [adjusted_rand_score(gt[valid].flatten(), l[valid].flatten())]

    return scores


class InstanceDiscrimination(nn.Module):

    def __init__(self, dim, n_instances, subsample=None, binary_dim=None):
        super().__init__()

        self.label_types = ['offset', 'boundaries', 'ids']

        self.subsample = subsample

        if binary_dim is not None:
            self.pred_head = nn.Parameter(torch.randn(dim, binary_dim), requires_grad=True)
            self.binary_classes = nn.Parameter(torch.rand(n_instances, binary_dim), requires_grad=False)
        else:
            self.pred_head = nn.Parameter(torch.randn(dim, n_instances), requires_grad=True)
            self.binary_classes = None

        self.predict_key = 'instances'

    # def example_sample(self, dataset_cls, split='train', bs=4):
    #     dataset = dataset_cls(split, [0], label_types=self.label_types)
    #     from torch.utils.data import DataLoader
    #     loader = DataLoader(dataset, batch_size=bs)
    #     return next(iter(loader))


    # def build_fake_output(self, sample):
    #     outputs = dict(centers=torch.cat([
    #         sample[self.predict_key][:, None].add(0.0001).mul(0.999).logit(),
    #     ], dim=1))
    #     return outputs

    def evaluate(self, model, dataset_val, bs=8, use_gt=False, n_workers=1):

        # nothing to evaluate, yet

        # device = next(model.parameters()).device
        # losses_val = []

        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=bs, shuffle=False, num_workers=1)

        model.eval()

        scores = []

        with torch.no_grad():
                
            for sample_val in loader_val:

                outputs = model(sample_val)          
                scores += rand_index(outputs, sample_val)

                # losses_val += [self.loss_function(outputs, sample_val)]
        
        out = dict(
            # loss = torch.tensor(losses_val).mean().item(),
            loss = 0,
            rand_index=torch.tensor(scores).mean().item()
        )

        return out


    def loss_function(self, outputs, labels, iteration=None):

        target_shape = labels['ids'].shape[1:4]
        device = outputs[self.predict_key].device

        if outputs[self.predict_key].shape[2:5] != target_shape:
            warnings.warn('output size does not match dataset size')
        
        out_scaled = torch.nn.functional.interpolate(outputs[self.predict_key], target_shape, mode='trilinear')
        out_scaled = out_scaled.permute(0,2,3,4,1).flatten(0, -2)

        if self.subsample is not None:
            indices = torch.multinomial(torch.ones(out_scaled.shape[0]), int(out_scaled.shape[0]*self.subsample))
        else:
            indices = slice(0, None)

        pred = out_scaled[indices] @ self.pred_head


        if self.binary_classes is not None:
            tgt = self.binary_classes[labels['ids']].flatten(0,-2)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, tgt[indices])
        else:
            loss = torch.nn.functional.cross_entropy(pred, labels['ids'].flatten()[indices].to(device))

        return loss, dict()


    def plot(self, sample, out=None, n=float('inf'), threshold=0):
        pass
        # no plot for now

        # n_cols = 2 if out is None else 3
        # bs = min(n, sample[self.predict_key].shape[0])
        # fig, ax = plt.subplots(bs, n_cols, figsize=(2*n_cols, 1.5*bs))
        
        # for i in range(bs):

        #     # image
        #     ax[i, 0].imshow(sample['image'][i,:,0].permute(1,2,0))

        #     # offsets in RGB
        #     ax[i, 1].imshow(sample[self.predict_key][i,0])

        #     if out is not None:
        #         ax[i, 2].imshow(out['centers'][i, 0,0].detach().cpu().sigmoid(), vmin=0, vmax=1)
  
        # [a.axis('off') for a in ax.flatten()]
        # fig.tight_layout()
        # plt.close(fig)
        # return fig
    
    
