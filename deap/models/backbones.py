import torch
import math
from torch import nn
from functools import partial
from deap.models import backbone_dict


class BackboneBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.feat_stride = None
        self.feat_dim = None
        self.precomputed_features = None

    def freeze(self):
        print('freeze backbone')
        for p in self.parameters():
            p.requires_grad = False

    def feature_dim(self):
        return self.feat_dim
    
    def precompute(self, loader, amp=True, dtype=None):
        """ this is the new precomputation function. """

        if self.precomputed_features is None:
            self.precomputed_features = dict()

        import tqdm
        device = next(self.parameters()).device
        for sample in tqdm.tqdm(loader):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                with torch.no_grad():
                    img = sample['image']
                    img = torch.nn.functional.interpolate(img[:,:,0], self.img_size, mode='bicubic', antialias=True)[:,:,None]
                    feats = self.forward(img.to(device))
            assert 'id' not in sample or isinstance(sample['id'][0], str)
            for f, key in zip(feats, sample['id']):
                if dtype is not None:
                    f = f.to(dtype)
                self.precomputed_features[key] = f.to(device)

    def features_compute_or_cached(self, x, keys, device):
        if self.precomputed_features is not None and all(k in self.precomputed_features for k in keys):
            return torch.stack([self.precomputed_features[k] for k in keys]).to(device)
        else:
            return self.forward(x)

    def base_size(self, img_size):
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        return img_size[0] // self.feat_stride, img_size[1] // self.feat_stride


class NoBackbone(BackboneBase):

    def __init__(self, feat_stride, feat_dim):
        super().__init__()
        self.feat_stride = feat_stride
        self.feat_dim = feat_dim



class TimmBackbone(BackboneBase):

    def __init__(self, model_name, feat_dim, feat_stride, img_size, cls_token='first', concat_layers=None, no_post=False, 
                 pretrained=True, normalize=False):
        super().__init__()
        import timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size, global_pool='')
        data_cfg = timm.data.resolve_data_config(self.backbone.pretrained_cfg)
        self.transform = timm.data.create_transform(**data_cfg)
        self.transform.transforms = self.transform.transforms[3:]

        self.feat_dim = feat_dim + (feat_dim*len(concat_layers) if concat_layers is not None else 0)
        self.feat_stride = feat_stride
        self.img_size = img_size if isinstance(img_size, (list, tuple)) else [img_size, img_size]
        self.cls_token = cls_token
        self.no_post = no_post
        self.concat_layers = concat_layers
        self.normalize = normalize

    def post_backbone(self, x, height, width):

        bs = x.shape[0]
        feat_dim = x.shape[2]
        s = {
            'first': lambda: slice(1, None), 
            'last': lambda: slice(0, -1), 
            'none': lambda: slice(0, None),
            'auto': lambda: slice(self.backbone.num_prefix_tokens, None)
        }[self.cls_token]()

        x = x.permute(0,2,1)[:,:,s]

        x = x.view(bs, feat_dim, height // self.feat_stride, width // self.feat_stride)
        
        if self.normalize:
            x = torch.nn.functional.normalize(x, 2, 1)

        return x
    
    def forward(self, x):

        if self.concat_layers is not None:
            extra_activations = dict()
            def hook(model, input, output, layer_i): 
                extra_activations[layer_i] = output.detach()

            # this is specific to DinoV2
            for layer_i in self.concat_layers:
                self.backbone.blocks[layer_i].register_forward_hook(partial(hook, layer_i=layer_i))
    
    
        bs, _, n_frames, height, width = x.shape
        x = x.permute(0,2,1,3,4).flatten(0,1)
        x = self.transform(x)
        x = self.backbone(x)

        if not self.no_post:
            x = self.post_backbone(x, height, width)

        if self.concat_layers is not None:
            x = torch.cat([x] + [self.post_backbone(extra_activations[l], height, width) 
                                 for l in self.concat_layers], dim=1)
        
        x = x.view(bs, n_frames, *x.shape[1:])
        x = x.transpose(1,2)
        return x



class TimmSwinBackbone(TimmBackbone):
    def post_backbone(self, x, height, width):

        bs = x.shape[0]
        feat_dim = x.shape[2]
        s = {
            'first': lambda: slice(1, None), 
            'last': lambda: slice(0, -1), 
            'none': lambda: slice(0, None),
            'auto': lambda: slice(self.backbone.num_prefix_tokens, None)
        }[self.cls_token]()

        return x.permute(0,3,1,2)
    


class TimmBackboneScalingOnScales(TimmBackbone):

    def __init__(self, model_name, feat_dim, feat_stride, img_size, cls_token='first', concat_layers=None, no_post=False, pretrained=True, highres=False):
        super().__init__(model_name, feat_dim, feat_stride, img_size, cls_token=cls_token, concat_layers=concat_layers, no_post=no_post, pretrained=pretrained)
        
        self.highres = highres
        fac = 2
        self.feat_dim = fac*feat_dim + (fac*feat_dim*len(concat_layers) if concat_layers is not None else 0)




    def base_size(self, img_size):
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        fac = 1 if self.highres else 2
        return img_size[0] // (fac*self.feat_stride), img_size[1] // (fac*self.feat_stride)

    def forward_backbone(self, x):

        bs, _, n_frames, height, width = x.shape
        x = x.permute(0,2,1,3,4).flatten(0,1)
        x = self.transform(x)
        x = self.backbone(x)

        if not self.no_post:
            x = self.post_backbone(x, height, width)        

        return x

    def cat_extra_activations(self, out, extra_activations):
        extra_act = torch.cat([a.clone() for a in extra_activations.values()], 2)
        extra_act = self.post_backbone(extra_act, 518, 518)
        return torch.cat([out, extra_act], dim=1)

    def forward(self, x):

        extra_activations = dict()

        if self.concat_layers is not None:
            
            def hook(model, input, output, layer_i): 
                extra_activations[layer_i] = output.detach()

            # this is specific to DinoV2
            for layer_i in self.concat_layers:
                self.backbone.blocks[layer_i].register_forward_hook(partial(hook, layer_i=layer_i))

        bs, _, n_frames, height, width = x.shape

        assert x.shape[3:] == (1036, 1036), f'actual size: {x.shape[3:]}'

        x_views = [x[:,:,:,i*518: (i+1)*518, j*518: (j+1)*518] for i in range(2) for j in range(2)]
        x_low = torch.nn.functional.interpolate(x, (1, 518, 518), mode='trilinear')

        x_views_ = []
        for x_ in x_views:

            out = self.forward_backbone(x_)

            if self.concat_layers is not None:
                out = self.cat_extra_activations(out, extra_activations)

            x_views_ += [out]
            
        x_views = x_views_
        # x_views = [self.forward_backbone(x_) for x_ in x_views]
        x_low = self.forward_backbone(x_low)
        if self.concat_layers is not None:
            x_low = self.cat_extra_activations(x_low, extra_activations)

        x_views = torch.cat([
            torch.cat([x_views[0], x_views[2]], dim=2),
            torch.cat([x_views[1], x_views[3]], dim=2),
        ], dim=3)

        if self.highres:
            x_low = torch.nn.functional.interpolate(x_low, (2*37, 2*37), mode='bilinear')
        else:
            x_views = torch.nn.functional.adaptive_avg_pool2d(x_views, (37, 37))
        
        x = torch.cat([x_low, x_views], dim=1)

        x = x.view(bs, n_frames, *x.shape[1:])
        x = x.transpose(1,2)
        return x
    

class TimmBackboneConcatCLS(TimmBackbone):

    def __init__(self, model_name, feat_dim, feat_stride, img_size, cls_token='first', no_post=False, pretrained=True):
        super().__init__(model_name, feat_dim, feat_stride, img_size, cls_token='first', no_post=False, pretrained=True)
        self.feat_dim = self.feat_dim*2

    def post_backbone(self, x, height, width):
        bs = x.shape[0]
        x = x.permute(0,2,1)

        x_no_cls = x[:,:,slice(self.backbone.num_prefix_tokens, None)]
        # x_cls = x[:,:,slice(0, self.backbone.num_prefix_tokens)]
        x_cls = x[:,:,0:1].repeat(1,1,x_no_cls.shape[2])  # extract and repeat cls token
        
        x = torch.cat([x_no_cls, x_cls], dim=1)

        return x.view(bs, self.feat_dim, height // self.feat_stride, width // self.feat_stride)
    

class TimmCNNBackbone(BackboneBase):

    def __init__(self, model_name, base_size, feat_dim, layers='last', pretrained=True, ):
        super().__init__()
        import timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        data_cfg = timm.data.resolve_data_config(self.backbone.pretrained_cfg)
        self.transform = timm.data.create_transform(**data_cfg)
        self.transform.transforms = self.transform.transforms[3:]
        self.layers = layers

        # self.feat_dim = feat_dim
        # self.feat_stride = feat_stride
        self.base_size_ = base_size

        self.feat_dim = feat_dim

    def base_size(self, s):
        return self.base_size_

    def forward(self, x):

        bs, _, n_frames, height, width = x.shape
        x = x.permute(0,2,1,3,4).flatten(0,1)
        x = self.transform(x)
        x = self.backbone(x)

        if self.layers == 'last':
            x = x[-1]
        elif self.layers == 'last_two':
            a = torch.nn.functional.interpolate(x[-1], (20, 20), mode='bilinear')
            x = torch.cat([x[-2], a], dim=1)

        x = x.view(bs, n_frames, *x.shape[1:])
        x = x.transpose(1,2)
        return x



class TimmAutoBackbone(BackboneBase):

    def __init__(self, model_name, no_post=False, pretrained=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        data_cfg = timm.data.resolve_data_config(self.backbone.pretrained_cfg)
        self.transform = timm.data.create_transform(**data_cfg)
        self.transform.transforms = self.transform.transforms[3:]

        self.feat_dim = self.backbone.num_features

        img_size = self.backbone.default_cfg['input_size']
        
        # figure out the stride
        o = self.backbone(torch.rand(1,*img_size))
        o = self.post_backbone(o, None, None)
        self.feat_stride = img_size[-1] // o.shape[-1]


        assert img_size[-2] == img_size[-1], 'currently only square input supported'
        self.img_size = img_size[-1]
        self.no_post = no_post

    def post_backbone(self, x, height, width):
        bs = x.shape[0]
        s = slice(self.backbone.num_prefix_tokens, None)
        x = x.permute(0,2,1)[:,:,s]

        out_size = math.isqrt(x.shape[-1])
        assert out_size**2 == x.shape[-1]
        return x.view(bs, self.feat_dim, out_size, out_size)
    
    def forward(self, x):

        bs, _, n_frames, height, width = x.shape
        x = x.permute(0,2,1,3,4).flatten(0,1)
        x = self.transform(x)
        x = self.backbone(x)

        if not self.no_post:
            x = self.post_backbone(x, height, width)
        
        x = x.view(bs, n_frames, *x.shape[1:])
        x = x.transpose(1,2)
        return x


class DinoBackbone(BackboneBase):

    def __init__(self, version='vit_s', img_size=224):
        super().__init__()
        import timm

        if version == 'vit_s':
            self.backbone = timm.create_model('vit_small_patch8_224.dino', pretrained=True, img_size=img_size, num_classes=0, global_pool='')
            self.feat_dim = 384
        elif version == 'vit_b':
            self.backbone = timm.create_model('vit_base_patch8_224.dino', pretrained=True, img_size=img_size, num_classes=0, global_pool='')
            self.feat_dim = 768
        else:
            raise ValueError('invalid version')
        
        data_cfg = timm.data.resolve_data_config(self.backbone.pretrained_cfg)
        # data_cfg = {k: v for k,v in data_cfg.items() if k in {'mean', 'std'}}
        self.transform = timm.data.create_transform(**data_cfg)
        self.transform.transforms = self.transform.transforms[3:]
        self.feat_stride = 8
        

    def post_backbone(self, x, height, width):
        bs = x.shape[0]
        return x.permute(0,2,1)[:,:,1:].view(bs, self.feat_dim, height // 8, width // 8)
    
    def feature_dim(self):
        return self.feat_dim
    
    def forward(self, x):

        bs, _, n_frames, height, width = x.shape
        x = x.permute(0,2,1,3,4).flatten(0,1)
        
        x = self.transform(x)
        # self.backbone.eval()
        x = self.backbone(x)
        x = self.post_backbone(x, height, width)

        x = x.view(bs, n_frames, *x.shape[1:])
        x = x.transpose(1,2)
        return x


class HieraBackbone(BackboneBase):

    def __init__(self):
        super().__init__()
        # sys.path.append('../../third_party/hiera/')
        # sys.path.append('third_party/hiera/')
        # import hiera
        # self.model = hiera.hiera_base_16x224(pretrained=True)
        self.model = torch.hub.load("facebookresearch/hiera", model="hiera_base_16x224", pretrained=True, checkpoint="mae_k400")
        # model(torch.randn(1,3,16,224,224), return_intermediates=True)   

        self.mean = nn.Parameter(torch.tensor([0.45, 0.45, 0.45]), requires_grad=False)     
        self.std = nn.Parameter(torch.tensor([0.225, 0.225, 0.255]), requires_grad=False)

    def feature_dim(self):
        return 768

    def base_size(self, img_size):
        return img_size // 32

    def forward(self, x):
        bs, _, n_frames, height, width = x.shape
        assert n_frames == 16

        x = x - self.mean.view(1, -1, 1, 1, 1)
        x = x / self.std.view(1, -1, 1, 1, 1)

        out = self.model(x, return_intermediates=True)
        # print(len(out))
        # print(out[1][3].shape)

        out = out[1][3]
        out = out.permute(0, 4, 1, 2, 3)

        return out


class ResNetDeepLab3Backbone(BackboneBase):

    def __init__(self):
        super().__init__()

        import torchvision
        from torchvision import transforms

        w = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', weights=w)
        self.inp_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.model.eval()
        self.feat_dim = 2048
        self.feat_stride = 8


    def forward(self, x):
        bs, _, n_frames, height, width = x.shape

        assert n_frames == 1
        x = x[:,:,0]

        x = self.inp_norm(x)

        out = self.model.backbone(x)['out']

        out = out[:,:,None]

        return out



class ResnetBackbone(BackboneBase):

    def __init__(self, version='resnet18', pretrained=False, output_layer=4):
        super().__init__()
        import timm

        print('RN pretrained', pretrained)

        self.backbone = timm.create_model(version, pretrained=pretrained, num_classes=0, global_pool='')
        data_cfg = timm.data.resolve_data_config(self.backbone.pretrained_cfg)
        # data_cfg = {k: v for k,v in data_cfg.items() if k in {'mean', 'std'}}
        self.transform = timm.data.create_transform(**data_cfg)
        self.transform.transforms = self.transform.transforms[3:]

        self.feat_dim = {'resnet18': 512, 'resnet50': 2048}[version]
        self.feat_dim = self.feat_dim // {4: 1, 3: 2, 2: 4}[output_layer]

        self.feat_stride = {4: 32, 3:16, 2:8}[output_layer]

        self.output_layer = output_layer
        

    # def feature_dim(self):
    #     return self.feat_dim
    
    # def base_size(self, img_size):
    #     return img_size // self.feat_stride

    def forward(self, x):

        bs, _, n_frames, height, width = x.shape
        x = x.permute(0,2,1,3,4).flatten(0,1)
        
        x = self.transform(x)
        # x = self.backbone(x)

        # resnet forward
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        
        if self.output_layer >= 2:
            x = self.backbone.layer2(x)
        
        if self.output_layer >= 3:
            x = self.backbone.layer3(x)
        
        if self.output_layer >= 4:
            x = self.backbone.layer4(x)

        x = x.view(bs, n_frames, *x.shape[1:])
        x = x.transpose(1,2)

        return x


class ResnetBackboneJoin(BackboneBase):

    def __init__(self, version='resnet18', pretrained=False):
        super().__init__()
        import timm

        print('RN pretrained', pretrained)
        

        self.backbone = timm.create_model(version, pretrained=pretrained, num_classes=0, global_pool='')
        data_cfg = timm.data.resolve_data_config(self.backbone.pretrained_cfg)
        # data_cfg = {k: v for k,v in data_cfg.items() if k in {'mean', 'std'}}
        self.transform = timm.data.create_transform(**data_cfg)
        self.transform.transforms = self.transform.transforms[3:]

        self.feat_dim = {'resnet18': 512, 'resnet50': 2048}[version]

        self.feat_dim = 256
        self.feat_stride = 8

        self.proj3 = nn.Conv2d(256, 128, kernel_size=1)
        self.proj4 = nn.Conv2d(512, 128, kernel_size=1)

        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(128*3, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )
        

    def forward(self, x):

        bs, _, n_frames, height, width = x.shape
        x = x.permute(0,2,1,3,4).flatten(0,1)
        
        x = self.transform(x)
        # x = self.backbone(x)

        # resnet forward
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        
        x2 = self.backbone.layer2(x)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        x3_up = nn.functional.interpolate(self.proj3(x3), x2.shape[-2:])
        x4_up = nn.functional.interpolate(self.proj4(x4), x2.shape[-2:])

        x = torch.cat([x2, x3_up, x4_up], dim=1)
        x = self.final(x)

        x = x.view(bs, n_frames, *x.shape[1:])
        x = x.transpose(1,2)

        return x



class MoGeBackbone(BackboneBase):

    def __init__(self):
        super().__init__()

        import sys
        import os

        current_file_path = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_file_path + '/../../../third_party/MoGe-main')
        sys.path.append(current_file_path + '/../../../third_party/utils3d')

        # from moge.model.v1 import MoGeModel
        from moge.model.v2 import MoGeModel # Let's try MoGe-2

        device = torch.device("cuda")

        self.feat_dim = 4
        self.feat_stride = 1

        # Load the model from huggingface hub (or load from local).
        # self.model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)        
        self.model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitb-normal").to(device)        
        

    def forward(self, x):
        x = x[:,:,0]
        # x = torch.nn.functional.interpolate(x, (100, 100), mode='bicubic', antialias=True)
        out = self.model.infer(x)
        out = torch.cat([out['depth'][:,:,:,None], out['normal']], dim=3)
        out = out.permute(0,3,1,2)[:,:,None]
        return out
        

class Aim2Backbone(BackboneBase):

    def __init__(self):
        super().__init__()

        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained("apple/aimv2-large-patch14-336")
        self.model = AutoModel.from_pretrained("apple/aimv2-large-patch14-336", trust_remote_code=True)
        self.feat_dim = 1024
        self.feat_stride = 14

    def forward(self, x):

        from PIL import Image

        images = [
            Image.fromarray(img.mul(255).byte().permute(1,2,0).numpy())
            for img in x[:,:,0].cpu()
        ]

        bs = len(images)
        inputs = self.processor(images=images, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].cuda()
        out = self.model(**inputs).last_hidden_state
        out = out.view(bs, 1, 24,24, 1024).permute(0,4,1,2,3)
        return out


class Phi35VBackbone(BackboneBase):

    def __init__(self):
        super().__init__()

        from transformers import AutoModelForCausalLM 
        from transformers import AutoProcessor

        from torchvision import transforms

        model_id = "microsoft/Phi-3.5-vision-instruct" 

        # Note: set _attn_implementation='eager' if you don't have flash_attn installed
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="cpu", 
            trust_remote_code=True, 
            torch_dtype="auto", 
            _attn_implementation='eager'    
        )
        self.vision_model = model.model.vision_embed_tokens.img_processor.vision_model
        self.vision_model.cuda()

        self.feat_dim = 1024
        self.feat_stride = 14

        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )

        self.resize = transforms.Resize((336,336))

        self.proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


    def forward(self, x):
        
        bs, _, n_frames, height, width = x.shape
        x = x.permute(0,2,1,3,4).flatten(0,1)

        x = self.normalize(x)
        x = self.resize(x)

        x = self.vision_model(x).last_hidden_state
        # x = x.view(bs, n_frames, *x.shape[1:])
        x = x[:,1:].view(bs, n_frames, 24, 24, 1024)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class Molmo7BDBackbone(BackboneBase):

    def __init__(self):
        super().__init__()

        from transformers import AutoModelForCausalLM, AutoProcessor
        from torchvision import transforms

        model_id = "allenai/Molmo-7B-D-0924" 

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        # Note: set _attn_implementation='eager' if you don't have flash_attn installed
        self.molmo_model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="cpu", 
            trust_remote_code=True, 
            torch_dtype="auto", 
            _attn_implementation='eager'    
        )
        self.vision_model = self.molmo_model.model.vision_backbone
        self.vision_model.cuda()

        self.feat_dim = 2048
        self.feat_stride = 14

        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )

        self.resize = transforms.Resize((336, 336))


    def preprocess(self, x):
        import utilities
        """ my attempt at implementing the preprocessing, didn't work """

        x = x.permute(0,2,1,3,4).flatten(0,1)

        x = utilities.pad_to_square(x)
        x = self.normalize(x)
        x = self.resize(x)

        patch_size = 14
        x = x.unfold(3, patch_size, patch_size).unfold(2, patch_size, patch_size)
        x = x.permute(0,2,3,1,4,5)
        x = x.flatten(3, 5).flatten(1,2)
        x = x[:,None]  
        return x      


    def forward(self, x):
        from PIL import Image
        
        bs, _, n_frames, height, width = x.shape
        assert n_frames == 1
        assert height == width == 336, 'this needs to be fixed because the preprocessor divides based on input size'

        images = [
            Image.fromarray(img.mul(255).byte().permute(1,2,0).numpy())
            for img in x[:,:,0].cpu()
        ]
        inputs = self.processor.process(images=images, text="")
        x = inputs['images'][::2]  # if image size is changed the 2 needs to be changed, too
        x = x[:, None].cuda()

        assert x.shape[0] == bs


        x = self.vision_model.encode_image(x)
        # x = x.view(bs, n_frames, *x.shape[1:])
        x = x[0].view(bs, n_frames, 24, 24, 2048)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class Qwen2_7BVLBackbone(BackboneBase):

    def __init__(self):
        super().__init__()

        from transformers import AutoProcessor
        from transformers import Qwen2VLForConditionalGeneration

        model_id = "Qwen/Qwen2-VL-7B-Instruct"

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        # Note: set _attn_implementation='eager' if you don't have flash_attn installed
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto") #, device_map="auto"

        self.vision_model = model.visual
        self.vision_model.merger = nn.Identity()

        self.vision_model.cuda()
        
        self.feat_dim = 3584 #1280
        self.feat_stride = 28 # 14

        a = torch.arange(144)
        a = torch.repeat_interleave(a, 4)
        a = a.unfold(0,4,4)
        a = a.view(12,12,2,2).permute(0,2,1,3)
        a = a.reshape(24,24)
        b = torch.arange(4).view(2,2).repeat(12,12)
        self.unseq24 = 4*a + b

    def forward(self, x):
        from PIL import Image
        
        bs, _, n_frames, height, width = x.shape
        assert n_frames == 1
        assert height == width == 336, 'this needs to be fixed because the preprocessor divides based on input size'

        images = [
            Image.fromarray(img.mul(255).byte().permute(1,2,0).numpy())
            for img in x[:,:,0].cpu()
        ]
        inputs = self.processor(images=images, text="", padding=True, return_tensors="pt",)

        assert x.shape[0] == bs

        inp = inputs['pixel_values'].cuda(), inputs['image_grid_thw'].cuda()
        torch.cuda.synchronize()
        x = self.vision_model(*inp)

        print(x.shape)

        # x = x.view(bs, 1, 12, 12, self.feat_dim)
        # x = x.permute(0,4,1,2,3)

        x = x.view(bs, 576, 1280)
    
        x = torch.vmap(lambda x:x[self.unseq24])(x)
        x = x.permute(0,3,2,1)[:,:,None]

    
        # x = x.unfold(0, output_size, output_size)
        #x = x.view(bs, 1, output_size, output_size, self.feat_dim)
        #x = x.permute(0, 4, 1, 2, 3)
        return x, inputs




backbone_dict.update({
    'vit_s-dino': partial(DinoBackbone, version='vit_s'),
    'vit_b-dino': partial(DinoBackbone, version='vit_b'),
    'vit_b-dino448': partial(DinoBackbone, version='vit_b', img_size=448),
    'vit_b-dino2reg': partial(TimmBackbone, 'vit_base_patch14_reg4_dinov2.lvd142m', 768, 14, 518, cls_token='auto'),
    'vit_b-dino2reg-L8-L10': partial(TimmBackbone, 'vit_base_patch14_reg4_dinov2.lvd142m', 768, 14, 518, cls_token='auto', concat_layers=[8, 10]),
    'vit_b-dino2reg-476-630': partial(TimmBackbone, 'vit_base_patch14_reg4_dinov2.lvd142m', 768, 14, (476, 630), cls_token='auto'),
    'vit_b-dino2reg-catcls': partial(TimmBackboneConcatCLS, 'vit_base_patch14_reg4_dinov2.lvd142m', 768, 14, 518, cls_token='auto'),    
    'vit_b-dino2reg336': partial(TimmBackbone, 'vit_base_patch14_reg4_dinov2.lvd142m', 768, 14, 336, cls_token='auto'),
    'vit_b-dino2': partial(TimmBackbone, 'vit_base_patch14_dinov2.lvd142m', 768, 14, 518, cls_token='auto'),
    'vit_l-dino2reg': partial(TimmBackbone, 'vit_large_patch14_reg4_dinov2.lvd142m', 1024, 14, 518, cls_token='auto'),
    'vit_l-dino2reg336': partial(TimmBackbone, 'vit_large_patch14_reg4_dinov2.lvd142m', 1024, 14, 336, cls_token='auto'),
    'vit_l-dino2reg-L8-L10': partial(TimmBackbone, 'vit_large_patch14_reg4_dinov2.lvd142m', 1024, 14, 518, cls_token='auto', concat_layers=[8, 10]),
    'vit_b-clip': partial(TimmBackbone, 'vit_base_patch32_clip_224.openai', 768, 32, 224),
    'vit_b16-clip': partial(TimmBackbone, 'vit_base_patch16_clip_224.openai', 768, 16, 224),
    'vit_l-clip336': partial(TimmBackbone, 'vit_large_patch14_clip_336.openai', 1024, 14, 336),
    'vit_b16-metaclip': partial(TimmBackbone, 'vit_base_patch16_clip_224.metaclip_2pt5b', 768, 16, 224),
    'vit_b-imagenet': partial(TimmBackbone, 'vit_base_patch16_224.orig_in21k', 768, 16, 224),
    'vit_l-imagenet': partial(TimmBackbone, 'vit_large_patch16_224.orig_in21k', 1024, 16, 224),
    'vit_b-imagenet1k': partial(TimmBackbone, 'vit_base_patch16_224.augreg_in1k', 768, 16, 224),
    'vit_b-siglip': partial(TimmBackbone, 'vit_base_patch16_siglip_384.webli', 768, 16, 384, cls_token='none'),
    'vit_b-siglip224': partial(TimmBackbone, 'vit_base_patch16_siglip_224.webli', 768, 16, 224, cls_token='none'),
    'vit_b-siglip512': partial(TimmBackbone, 'vit_base_patch16_siglip_512.webli', 768, 16, 512, cls_token='none'),
    'vit_b-siglip_so': partial(TimmBackbone, 'vit_so400m_patch14_siglip_384', 1152, 14, 512, cls_token='none'),
    'vit_b-mae': partial(TimmBackbone, 'vit_base_patch16_224.mae', 768, 16, 224),
    'vit_l-mae': partial(TimmBackbone, 'vit_large_patch16_224.mae', 1024, 16, 224),
    'vit_b-mae448': partial(TimmBackbone, 'vit_base_patch16_224.mae', 768, 16, 448),
    'hiera_b-mae': partial(TimmBackbone, 'hiera_base_224.mae', 768, 32, 224, cls_token='none'),
    'hiera_bplus-mae': partial(TimmBackbone, 'hiera_base_plus_224.mae', 896, 32, 224, cls_token='none'),
    'vit_b-mae-untrained': partial(TimmBackbone, 'vit_base_patch16_224.mae', 768, 16, 224, pretrained=False),

    #'swin_l-': partial(TimmBackbone, 'swinv2_large_window12to24_192to384.ms_in22k_ft_in1k', 1024, 16, 224),

    'phi35v': Phi35VBackbone,
    'molmo7bd': Molmo7BDBackbone,
    'qwen27bvl': Qwen2_7BVLBackbone,


    'convnext_laion_soup320': partial(TimmCNNBackbone, 'convnext_large_mlp.clip_laion2b_ft_soup_320', base_size=(10, 10), feat_dim=1536),
    'convnext_laion_soup320_last2': partial(TimmCNNBackbone, 'convnext_large_mlp.clip_laion2b_ft_soup_320', base_size=(20, 20), layers='last_two', feat_dim=2304),
    'vitamin_l2_384': partial(TimmBackbone, 'vitamin_large2_384', 1024, 16, 384, cls_token='none'),
    'aim2_l14': partial(Aim2Backbone, ),

    'hiera_b': HieraBackbone,
    'resnet50_deeplab': partial(ResNetDeepLab3Backbone),
    'resnet18': partial(ResnetBackbone, version='resnet18'),
    'resnet18j': partial(ResnetBackboneJoin, version='resnet18'),
    'resnet18-l3': partial(ResnetBackbone, version='resnet18', output_layer=3),
    'resnet18-l3-in': partial(ResnetBackbone, version='resnet18', pretrained=True, output_layer=3),
    'resnet18-l2': partial(ResnetBackbone, version='resnet18', output_layer=2),
    'resnet50': partial(ResnetBackbone, version='resnet50'),
})
