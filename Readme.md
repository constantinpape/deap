# Dense Attentive Probing


[[TMLR Paper](https://openreview.net/forum?id=neMAx4uBlh)] [[Project page](https://eckerlab/projects/deap)]

In this repository we provide the source code of dense attentive probing.

## Preparation

We require only a few dependencies. Install them using:

`pip install timm torchvision scikit-learn pillow`


Set the following environment variables:
* `PASCAL_VOC_PATH`: Pascal VOC 2012 folder. Uses the torchvision dataset `VOCSegmentation`.
* `COCO_PATH`: base folder of COCO 2017. Download from [here](https://cocodataset.org/#download).
* `COCO_STUFF_PATH`: base foldeer of COCO Stuff. Download [here](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip)
* `NYU_DEPTH_PATH`: base folder of NYUv2 depth. Download from huggingface: `hf_sayakpaul_nyu_depth_v2`.


## Experiments

All experiment configurations are described in `experiments.py`.

```python
from train import train, evaluate
from experiments import semseg_runs

train(semseg_runs[0])
```
This will save model weights, e.g. `clip-weights.pth`. These can be evaluated using

```python
evaluate(semseg_runs[0], 'clip-weights.pth')
```


## Add new backbones

New backbones can be implemented in a straightforward way. If the backbone is a vision transformer supported by timm you can use `TimmBackbone`, for example like this:

```python
from functools import partial
MetaCLIPBackbone = partial(TimmBackbone, 'vit_base_patch16_clip_224.metaclip_2pt5b', feat_dim=768, feat_stride=16, img_size=224)
```

To use this in the model, add this to your python script:
```python
from deap.models import backbone_dict
backbone_dict.update(dict(my_metaclip_backbone=MetaCLIPBackbone))
```


## Model Usage

```python
model = SelfAttReadouts(
    base_size=28, # base size for the queries 
    up=(2,2,2),  # up-scale CNN config. In this case 3 layers with a factor of 2 each.
    decoder='CA-A3-sl', 
    dim=16, 
    inp_img_size=518, 
    backbone_name='vit_b-dino2reg',  
    outputs=[('output', 1)]
)
model(dict(image=torch.rand(1,3,1,518,518)))
```