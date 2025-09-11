from deap.xdict import xdict


# base training config
default = xdict(
    lr=0.001,
    n_iterations=60,
    n_workers=4,
    log_dir='logs',
    no_wandb=True,
    amp=True,
    bs=32,
    wd=0.01,
    val_interval=250,    
)


# MODELS

model_cfg = xdict(
  __class__='deap.models.dense_attentive_probing.SelfAttReadouts',
  base_size=56,
  decoder='CA-A3-sl',
  dim=16,
  init_sincos=True,
  inp_img_size=224,
  n_heads=8,
  up=(2,2,2),
)

backbones = ['vit_b-mae-untrained', 'vit_b-imagenet', 'vit_b-mae', 'hiera_bplus-mae', 'vit_b16-clip']

# baselines
model_cfg_conv = xdict(inp_img_size=518, dim=48, decoder='C', backbone_name='vit_b-dino2reg')
model_cfg_featup = xdict(__class__ = 'deap.models.dense_attentive_probing.SelfAttReadoutsFeatup', inp_img_size=280)
model_cfg_bil = xdict(inp_img_size=518, dim=48, decoder='BI-ff', backbone_name='vit_b-dino2reg')


# SEMANTIC SEGMENTATION

semseg_dataset = xdict(
    __class__ = 'deap.datasets.pascal_simple.PascalVOC12Segmentation',
    img_size = 224,
)
semseg = xdict(
    task=xdict(
      __class__='deap.tasks.semseg.SemanticSegmentationTask',
      ignore_index=255
    ),
    dataset=xdict(split='train') + semseg_dataset,
    dataset_val=xdict(split='trainval', max_samples=500) + semseg_dataset,
    dataset_test=xdict(split='val', max_samples=500) + semseg_dataset,
    model=xdict(outputs=(('sem', 21),),) + model_cfg,
)  + default

semseg_runs = [xdict(name=f'sem_{b}', model=xdict(backbone_name=b)) + semseg for b in backbones]


# COCO STUFF

coco_stuff_dataset = xdict(
    __class__ = 'deap.datasets.coco_simple.COCOStuff',
    img_size = 224,
)
coco_stuff_cfg = xdict(
    lr=0.001,
    n_iterations=20000,
    task=xdict(
        __class__= 'deap.tasks.semseg.SemanticSegmentationTask',
        ignore_index = 255,  # remove for VIPSeg
        n_classes = 200,
    ),
    dataset=xdict(split='train') + coco_stuff_dataset,
    dataset_val=xdict(split='val', max_samples=500) + coco_stuff_dataset,
    dataset_test=xdict(split='test') + coco_stuff_dataset,
    model=xdict(base_size=28, up=(2, 2, 2), dim_interm=32, outputs=(('sem', 200),),) + model_cfg,
) + default

coco_stuff_runs = [xdict(name=f'sem_{b}', model=xdict(backbone_name=b)) + coco_stuff_cfg for b in backbones]


# DEPTH

depth_dataset = xdict(
    __class__ = 'deap.datasets.nyu_depth_simple.NYUDepthV2',
    img_size = (216, 288),  # to cover the full image for fair comparison, will be scaled to backbone sizes
)

depth_cfg = xdict(
    lr=0.001, 
    n_iterations=3000,
    val_interval=100,
    n_workers=8,
    task=xdict(__class__= 'deap.tasks.depth_estimation.DepthPredictionTaskSigloss'),
    dataset=xdict(split='train', aug=()) + depth_dataset,
    dataset_val=xdict(split='val2', img_size=(480, 640), aug=()) + depth_dataset,
    dataset_test=xdict(split='test', img_size=(480, 640), aug=()) + depth_dataset,
    model=xdict(base_size=28, up=(2, 2, 2), out_bias=1, outputs=(('depth', 1),)) + model_cfg,
) + default

depth_runs = [xdict(name=f'depth_{b}', model=xdict(backbone_name=b)) + depth_cfg for b in backbones]


# BOUNDARY

boundary_dataset = xdict(
    __class__ = 'deap.datasets.coco_simple.COCO',
    label_types=['ids', 'boundaries'],
    img_size=448,
    inp_img_size=224,
)
boundary_cfg = xdict(
    lr=0.001, 
    n_iterations=10000,
    n_workers=16,
    task=xdict(__class__= 'deap.tasks.scalar_maps.BoundaryPredictionTask'),
    dataset=xdict(split='train', max_samples=5000) + boundary_dataset,
    dataset_val=xdict(split='val', max_samples=500) + boundary_dataset,
    model=xdict(base_size=56, up=(2, 2, 2), outputs=(('boundaries', 1),)) + model_cfg,
) + default

boundaries_runs = [xdict(name=f'depth_{b}', model=xdict(backbone_name=b)) + boundary_cfg for b in backbones]

# CENTERNET

centernet_dataset = xdict(
    __class__ = 'deap.datasets.coco_simple.COCO',
    label_types=['centers_gauss'],
    img_size=448,
    inp_img_size=224,
)

centernet_cfg = xdict(
    lr=0.001, 
    wandb_project='deap',
    n_iterations=20000,
    val_interval=1000,
    n_workers=16,
    task=xdict( __class__= 'deap.tasks.center_net.CenterNetTask'),
    dataset=xdict(split='train', aug=False) + centernet_dataset,
    dataset_val=xdict(split='val', aug=False, max_samples=500) + centernet_dataset,
    dataset_test=xdict(split='test', aug=False, max_samples=500) + centernet_dataset,  
    model=xdict(base_size=56, up=(2, 2, 2), outputs=(('centers_gauss', 1), ('centers_size', 2)))  + model_cfg,
) + default

centernet_runs = [xdict(name=f'depth_{b}', model=xdict(backbone_name=b)) + centernet_cfg for b in backbones]