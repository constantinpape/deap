import torch
import math
from torch import nn
from functools import partial


DECODER_CLASSES = dict()



def pos_enc_sincos(x, dim=10, flatten=False, temperature=10000.0):
    """ 
    sinosoidal positional encoding 
    important: value range of x must be considered. Works best for integers, i.e. torch.arange(N).
    """

    old_shape = x.shape
    x = x.flatten()
    # div_vec = 1 / (10000 ** (2 * torch.arange(dim).float() / dim))
    div_vec = torch.exp(torch.arange(0, dim, 2).to(x.device) * (-math.log(temperature) / dim))

    pe2 = torch.cat([
        torch.sin(x[:,None] * div_vec[None, :]),
        torch.cos(x[:,None] * div_vec[None, :]),
    ], dim=1)

    pe2 = pe2.view(old_shape + (dim,))

    if flatten:
        pe2 = pe2.flatten(-2, -1)
        
    return pe2


def init_queries_sincos2(base_size, pe_dim, dim, fac=1):
    pos = torch.dstack(torch.meshgrid(torch.arange(base_size[1]), torch.arange(base_size[0]), indexing='xy'))
    return torch.cat([
        pos_enc_sincos(pos*fac, dim=pe_dim // 2, flatten=True),
        torch.randn(base_size[0],base_size[1], dim)
    ], dim=2).flatten(0, 1)


def gen_mask(x, q, mask_thresh, n_heads):
    if mask_thresh is not None:
        bs = x.shape[0]
        m = get_distances(x, q) < mask_thresh
        m = m[None].repeat(bs, 1, 1).to(q.device)    
    else:
        m = [None]*n_heads

    return m

def square_meshgrid(size, indexing='xy', device='cpu'):
    return torch.dstack(
        torch.meshgrid(torch.linspace(0, 1, size, device=device), 
                       torch.linspace(0, 1, size, device=device), indexing=indexing)
    )


def rect_meshgrid(sizes, device='cpu'):
    return torch.dstack(
        torch.meshgrid(torch.linspace(0, 1, sizes[0], device=device), 
                       torch.linspace(0, 1, sizes[1], device=device), indexing='xy')
    )

def get_distances(x, q):
    device = x.device
    s1, s2 = int(math.sqrt(x.shape[1])), int(math.sqrt(q.shape[1]))
    m1, m2 = square_meshgrid(s1, device=device).flatten(0,1), square_meshgrid(s2, device=device).flatten(0,1)
    return (m2[:, None] - m1[None, :]).pow(2).sum(2) 


def get_distances_rect(s1, s2, device='cpu'):
    # permute(1,0,2) on s2?
    m1, m2 = rect_meshgrid(s1, device=device).flatten(0,1), rect_meshgrid(s2, device=device).flatten(0,1)
    return (m2[:, None] - m1[None, :]).pow(2).sum(2) 


def gaussian(d, sigma):
    return torch.exp(-0.5 * (d / sigma) ** 2) / (sigma * torch.sqrt(torch.tensor(2.0 * torch.pi)))

def rbf_kernel(d, gamma):
    return torch.exp(-gamma * d.pow(1))

class DecoderCA(nn.Module):

    def __init__(self, parent, n_classes, dec_cls, mlp_cls, gn=False):
        super().__init__()
        p = parent
        
        self.img_size = p.inp_img_size
        self.base_size = p.base_size
        self.dim = p.dim
        self.n_classes = n_classes

        self.gn = nn.GroupNorm(16, p.inp_dim) if gn else nn.Identity()

        self.cross_att = dec_cls(parent)
        self.mlp = mlp_cls(p, n_classes)

        from deap import utilities as ut
        print(f'decoder params: {ut.count_parameters(self)/1e6:.3f}M')        
        print(f'recon: {ut.count_parameters(self.mlp)/1e6:.3f}M')  
        print(f'CA: {ut.count_parameters(self.cross_att)/1e6:.3f}M')  

    def forward(self, feats):

        bs, _, n_frames = feats.shape[:3]

        feats = self.gn(feats)

        # print(self.dim)
        out = self.cross_att(feats)
        out = out.permute(0, 2, 1).view(bs, self.dim, 1, self.base_size[0], self.base_size[1])
        # out = out.permute(0,1,2,3,4)
        out = self.mlp(out)

        return out



class Interp(nn.Module):

    def __init__(self, p, n_classes):
        super().__init__()
        dim_interm = max(p.dim // 4, n_classes // 2) if p.dim_interm is None else p.dim_interm
        out_mlp = []
        for i, s in enumerate(p.up):
            dim_in = p.dim if i==0 else dim_interm
            out_mlp += [
                nn.Conv3d(dim_in, dim_interm, kernel_size=(1,3,3), padding=(0,1,1)), nn.ReLU(), 
                nn.ConvTranspose3d(dim_interm, dim_interm, kernel_size=(1, s, s), stride=(1, s, s)), nn.ReLU()
            ]
                
        out_mlp += [nn.Conv3d(dim_interm, n_classes, kernel_size=1)]
        self.out_mlp = nn.Sequential(*out_mlp)
        self.skip_mlp = nn.Sequential(
            InterpolateFac(torch.prod(torch.tensor(p.up)).item(), bilinear=True),
            nn.Conv3d(p.dim, n_classes, kernel_size=1),
        )

        if hasattr(p, 'out_bias') and p.out_bias is not None:
            self.skip_mlp[1].bias = nn.Parameter(self.skip_mlp[1].bias+p.out_bias)
            self.out_mlp[-1].bias = nn.Parameter(self.out_mlp[-1].bias+p.out_bias)

    def forward(self, out):
        out = self.skip_mlp(out) + self.out_mlp(out)
        return out


class BilinearInterp(nn.Module):

    def __init__(self, p, n_classes):
        super().__init__()

        self.fac = torch.prod(torch.tensor(p.up))
        self.interp = InterpolateFac(self.fac, bilinear=True)
        self.proj = nn.Conv3d(p.dim, n_classes, kernel_size=(1,3,3), padding=(0,1,1))

    def forward(self, out):
        out = self.proj(out)
        out = self.interp(out)

        return out

DecCAInterp = partial(DecoderCA, mlp_cls=Interp)


class CrossAtt3(nn.Module):
     
    def __init__(self, p, config=[(0.05, True), (None, False)], layer_cls=None, layer_args=None):
        super().__init__()

        if layer_cls is None:
            layer_cls = CrossAttLayer3

        self.layer_cls = layer_cls
        if isinstance(layer_cls, str):
            layer_cls = globals()[layer_cls]
        self.layer_args = layer_args if layer_args is not None else dict()
        self.init_layers(p, config)

        # this used to be 0*init_queries_sincos2(p.base_size, p.pe_dim, 0) for a while
        self.queries = nn.Parameter(init_queries_sincos2(p.base_size, p.pe_dim, 0), requires_grad=False)
 
        if p.feats_pe:
            self.feats_pe = nn.Parameter(init_queries_sincos2(p.f_base_size, p.pe_dim, 0), requires_grad=False)
        else:
            self.feats_pe = None

        self.no_res = True


    def init_layers(self, p, config):
        self.layers = nn.ModuleList([
            self.layer_cls(p, is_first=i_layer==0, sigma_per_head=sigma_per_head, sigma_init=sigma, **self.layer_args)
            for i_layer, (sigma, sigma_per_head) in enumerate(config)
        ])

    def forward(self, feats):
        bs = feats.shape[0]
    
        feats = feats.flatten(2).permute(0,2,1)

        if self.feats_pe is not None:
            feats_pe = self.feats_pe[None].repeat(bs, 1, 1, 1, 1).flatten(1,3)
            # print(feats.shape, feats_pe.shape)
            feats = torch.cat([feats, feats_pe], dim=2)
        
        x = self.queries
        for j, layer in enumerate(self.layers):
            if self.no_res:
                x = layer(x, feats)
            else:
                x = x + layer(x, feats)

        return x


class CrossAttNew(CrossAtt3):

    def __init__(self, p, config=None, n_layers=1, layer_cls=None, layer_args=None):
        self.n_layers = n_layers
        super().__init__(p, config=config, layer_args=layer_args, layer_cls=layer_cls)
        

    def init_layers(self, p, config):
        self.layers = nn.ModuleList([
            self.layer_cls(p, is_first=i_layer==0, **self.layer_args)
            for i_layer in range(self.n_layers)
        ])


class CrossAtt3NoFF(CrossAtt3):

    def init_layers(self, p, config):
        self.layers = nn.ModuleList([
            CrossAttLayer3(p, is_first=i_layer==0, sigma_per_head=sigma_per_head, sigma_init=sigma, only_mask=False, no_ff=True)
            for i_layer, (sigma, sigma_per_head) in enumerate(config)
        ])        

class CrossAtt3VF1(CrossAtt3):

    def init_layers(self, p, config):
        self.layers = nn.ModuleList([
            CrossAttLayer3(p, is_first=i_layer==0, sigma_per_head=sigma_per_head, sigma_init=sigma, vdim_fac=1, only_mask=False, no_ff=True)
            for i_layer, (sigma, sigma_per_head) in enumerate(config)
        ])

class CrossAtt3VF1OnlyMask(CrossAtt3):

    def init_layers(self, p, config):
        self.layers = nn.ModuleList([
            CrossAttLayer3(p, is_first=i_layer==0, sigma_per_head=sigma_per_head, sigma_init=sigma, vdim_fac=1, only_mask=True, no_ff=True)
            for i_layer, (sigma, sigma_per_head) in enumerate(config)
        ])

class CrossAtt3OnlyMask(CrossAtt3):

    def init_layers(self, p, config):
        self.layers = nn.ModuleList([
            CrossAttLayer3(p, is_first=i_layer==0, sigma_per_head=sigma_per_head, sigma_init=sigma, only_mask=True)
            for i_layer, (sigma, sigma_per_head) in enumerate(config)
        ])



class CrossAttLayer3(nn.Module):

    def __init__(self, p, is_first=False, vdim_fac=2, 
                 mask_thresh=None, sigma_per_head=False, sigma_init=None, 
                 dropout=0, dim_out=None, only_mask=False, no_ff=False):
        super().__init__()

        n_heads, dim, dim_inp = p.n_heads, p.dim // p.n_heads, p.inp_dim + 8

        self.n_heads = n_heads
        self.dim = dim
        self.mask_thresh = mask_thresh
        self.only_mask = only_mask
        self.vdim_fac = vdim_fac
        # self.q = nn.ModuleList([torch.randn(1, 10, 128) for _ in range(n_heads)])

        self.feat_token_shape = torch.tensor(p.inp_img_size) // p.backbone.feat_stride
        self.base_size = p.base_size
        # self.out_bias = p.out_bias

        if is_first:
            print('first')
            self.proj_q = nn.Linear(p.pe_dim, dim*n_heads)
        else:
            self.proj_q = nn.Identity()

        if sigma_init is not None:
            sig = torch.rand(self.n_heads) if sigma_per_head else torch.rand(1)
            if isinstance(sigma_init, float):
                sig.fill_(sigma_init)
            self.sigma = nn.Parameter(sig)
        else:
            self.sigma = None

        self.proj_k = nn.ModuleList([nn.Linear(dim_inp, dim) for _ in range(n_heads)])
        self.proj_v = nn.ModuleList([nn.Linear(dim_inp, vdim_fac*dim) for _ in range(n_heads)])
        self.norm_inp = nn.LayerNorm(dim_inp, eps=1e-5, bias=True)

        if only_mask:
            for layer in self.proj_k:
                for p in layer.parameters():
                    p.requires_grad = False
                layer.weight.fill_(0)
                layer.bias.fill_(1)

        dim_out = dim_out if dim_out is not None else n_heads*dim

        if not no_ff:
            self.norm2 = nn.LayerNorm(dim_out, eps=1e-5, bias=True)
            dim_feedforward = vdim_fac*2*n_heads*dim
            self.linear1 = nn.Linear(vdim_fac*n_heads*dim, dim_feedforward, bias=True)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, dim_out, bias=True)
        else:
            self.linear1 = None
            self.proj2 = nn.Linear(vdim_fac*n_heads*dim, dim_out)

    def forward(self, q, feats):

        # x = self.norm_inp(x)
        bs = feats.shape[0]

        if torch.rand(1).item() < 0.01:
            # print('sigma', self.sigma.mean(), id(self))
            print('sigma', self.sigma)
            # import ipdb; ipdb.set_trace()

        q = self.proj_q(q)

            # q = q.repeat(1, self.n_heads)
        if q.ndim == 2:
            q = q[None].repeat(bs, 1, 1)

        if self.sigma is None:
            m = gen_mask(feats, q, self.mask_thresh, self.n_heads)
        else:
            # d = get_distances(feats, q)
            d = get_distances_rect(
                (self.feat_token_shape[1], self.feat_token_shape[0]), 
                (self.base_size[1], self.base_size[0]), 
                device=feats.device
            )
            # d = get_distances_rect(self.base_size, self.feat_token_shape, device=feats.device)
            # m = rbf_kernel(d, self.sigma).log()
            # print(d.shape)

            if len(self.sigma) == 1:
                m = gaussian(d, self.sigma)
                m = [m for _ in range(self.n_heads)]
            else:
                m = [gaussian(d, s) for s in self.sigma]

        if self.only_mask:
            # fill key and queries with ones --> remove information
            q.fill_(1)

        x = torch.cat([
            nn.functional.scaled_dot_product_attention(
                q[..., i*self.dim:(i+1)*self.dim],
                self.proj_k[i](feats),
                self.proj_v[i](feats),
                attn_mask=m[i]
            ) for i in range(self.n_heads)
        ], dim=2)

        if self.linear1 is not None:
            x = self.norm2(self.linear2(self.dropout(nn.functional.relu(self.linear1(x)))))
        else:
            x = self.proj2(x)

        return x



DECODER_CLASSES.update({
    'CA-A3-sl': partial(DecoderCA, mlp_cls=Interp, dec_cls=partial(CrossAtt3, config=[(0.05, True)])),
    'CA-A3-sl-vf1': partial(DecoderCA, mlp_cls=Interp, dec_cls=partial(CrossAtt3VF1, config=[(0.05, True)])),
    'CA-A3-sl-vf1-om': partial(DecoderCA, mlp_cls=Interp, dec_cls=partial(CrossAtt3VF1OnlyMask, config=[(0.05, True)])),
    'CA-A3-d2': partial(DecoderCA, mlp_cls=Interp, dec_cls=partial(CrossAtt3, config=[(0.05, True), (1.0, False)])),
    'CA-A3-d22': partial(DecoderCA, mlp_cls=Interp, dec_cls=partial(CrossAtt3, config=[(0.05, True), (0.05, True)])),
    'CA-A3-sl-noff': partial(DecoderCA, mlp_cls=Interp, dec_cls=partial(CrossAtt3NoFF, config=[(0.05, True)])),    

    'CA-A3-sl-no-indiv-sigma': partial(DecoderCA, mlp_cls=Interp, dec_cls=partial(CrossAtt3, config=[(0.05, False)])),
    'CA-A3-sl-no-sigma': partial(DecoderCA, mlp_cls=Interp, dec_cls=partial(CrossAtt3, config=[(None, True)])),
    'CA-A3-sl-only-mask': partial(DecoderCA, mlp_cls=Interp, dec_cls=partial(CrossAtt3OnlyMask, config=[(0.05, True)])),
    'CA-A3-sl-bilinear': partial(DecoderCA, mlp_cls=BilinearInterp, dec_cls=partial(CrossAtt3, config=[(0.05, True)])),    
    'CA-A3-sl-no-ff': partial(DecoderCA, mlp_cls=Interp, dec_cls=partial(CrossAtt3NoFF, config=[(0.05, True)])),    
})





class BicubicAttention(nn.Module):

    def __init__(self, p, mode='bicubic'):
        super().__init__()
        h, w = p.base_size
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
        self.grid = nn.Parameter(torch.stack([grid_x, grid_y], dim=-1), requires_grad=False)
        self.mode = mode

    def forward(self, feats_low, feats):
        bs = feats_low.shape[0]
        out = torch.nn.functional.grid_sample(feats_low, self.grid[None].repeat(bs, 1, 1, 1), mode=self.mode, align_corners=True)
        out = out.flatten(2).permute(0,2,1)
        return out, None
    



class AttInter(nn.Module):

    def __init__(self, p, att_cls, ff_cls):
        super().__init__()

        # assert p.base_size[0]==p.base_size[1]
        # assert p.f_base_size[0]==p.f_base_size[1]
        assert p.dim % p.n_heads == 0

        self.att = att_cls(p)
        self.ff = ff_cls(p)
        self.n_heads = p.n_heads
        self.proj = nn.Conv3d(p.inp_dim, p.dim, kernel_size=1)


    def forward(self, feats, with_att=False):
        # bs = feats.shape[0]

        feats_low = self.proj(feats)
        assert feats_low.shape[2] == 1
        assert feats.shape[2] == 1
        feats_low = feats_low[:,:,0]
        feats = feats[:,:,0]

        out, att = self.att(feats_low, feats)
        out = self.ff(out)
        
        if with_att:
            return out, att
        else:
            return out


class FF(nn.Module):

    def __init__(self, p, dropout=0):
        super().__init__()

        dim_in = dim_out = p.dim
        dim_feedforward = 2*p.dim

        self.norm2 = nn.LayerNorm(dim_out, eps=1e-5, bias=True)
        self.linear1 = nn.Linear(dim_in, dim_feedforward, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_out, bias=True)

    def forward(self, x):
        x = self.norm2(self.linear2(self.dropout(nn.functional.relu(self.linear1(x)))))
        return x


class NoFF(nn.Module):

    def __init__(self, p, dropout=0):
        super().__init__()

    def forward(self, x):
        return x


class BilinearAttention(BicubicAttention):
    def __init__(self, p):
        super().__init__(p, mode='bilinear')

DECODER_CLASSES.update({
    'BI': partial(DecoderCA, mlp_cls=Interp, dec_cls=partial(AttInter, att_cls=BicubicAttention, ff_cls=NoFF)),
    'BIL': partial(DecoderCA, mlp_cls=Interp, dec_cls=partial(AttInter, att_cls=BilinearAttention, ff_cls=NoFF)),
    'BI-ff': partial(DecoderCA, mlp_cls=Interp, dec_cls=partial(AttInter, att_cls=BicubicAttention, ff_cls=FF)),
})


class Interpolation(nn.Module):

    def __init__(self, p, mode='bicubic'):
        super().__init__()
        self.proj = nn.Conv3d(p.inp_dim, p.dim, kernel_size=1)
        h, w = p.base_size
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
        self.grid = nn.Parameter(torch.stack([grid_x, grid_y], dim=-1), requires_grad=False)
        self.mode = mode


    def forward(self, feats):
        bs = feats.shape[0]

        feats = self.proj(feats)
        assert feats.shape[2] == 1
        # torch.Size([1, 768, 1, 14, 14])
        feats = feats[:,:,0]
        
        out = torch.nn.functional.grid_sample(feats, self.grid[None].repeat(bs, 1, 1, 1), mode=self.mode, align_corners=True)

        out = out.flatten(2).permute(0,2,1)

        return out


class InterpolateFac(nn.Module):

    def __init__(self, fac, bilinear=False):
        super().__init__()
        self.fac = fac
        self.interp_args = dict()
        if bilinear:
            self.interp_args = dict(mode='bilinear', antialias=True)

    def forward(self, x):
        
        size = x.shape[3:]
        assert x.shape[2] == 1

        size_new = [s*self.fac for s in size]
        return torch.nn.functional.interpolate(x[:,:,0], size_new, **self.interp_args)[:,:,None]



class ConvDecoder(nn.Module):

    def __init__(self, parent, n_classes, depth=1, no_conv=False, no_gn=False):
        super().__init__()
        p = parent
        
        self.img_size = p.inp_img_size
        self.base_size = p.base_size
        self.dim = p.dim

        dim2 = max(p.dim // 6, n_classes * 3)

        n_frames = 1

        if no_gn:
            self.gn = self.gn2 = nn.Identity()
        else:
            self.gn = nn.GroupNorm(16, p.inp_dim)   
            self.gn2 = nn.GroupNorm(8, p.dim + 8)        

        # self.proj_q = nn.Linear(32 + 16, self.dim)
        self.proj_feats = nn.Conv3d(p.inp_dim, self.dim, kernel_size=1)

        ks = max(1, p.f_base_size[0] // self.base_size[0]), max(1, p.f_base_size[1] // self.base_size[1])
        
        if no_conv:
            self.conv = nn.Identity()
        else:
            self.conv = nn.Conv3d(self.dim, self.dim, kernel_size=(1,ks[0],ks[1]), stride=(1,ks[0],ks[1]))        

        # TODO: reduce dim of PEs to 16 or smaller. self.dim is probably way too much.

        # if not p.init_sincos:
        #     self.feats_pe = nn.Parameter(3*torch.randn(n_frames, *self.base_size, 8))
        # else:
        self.feats_pe = nn.Parameter(init_queries_sincos2(self.base_size, 8, 0).view(*self.base_size, 8))

        self.self_att = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(p.dim + 8, p.n_heads, 4*p.dim, batch_first=True, dropout=0.0), depth)

        # self.net1 = nn.Sequential(
        #     # nn.Conv3d(p.dim, p.dim, kernel_size=(1,3,3), padding=(0,1,1)), nn.ReLU(),
        #     nn.ConvTranspose3d(p.dim + 8, dim2, kernel_size=(1,p.up[0],p.up[0]), stride=(1,p.up[0],p.up[0])), nn.ReLU(),
        #     nn.Conv3d(dim2, dim2, kernel_size=(1,3,3), padding=(0,1,1)), nn.ReLU(), 
        #     nn.ConvTranspose3d(dim2, dim2, kernel_size=(1,p.up[1], p.up[1]), stride=(1, p.up[1], p.up[1])), nn.ReLU(),
        #     nn.Conv3d(dim2, n_classes, kernel_size=1)
        # )     

        from types import SimpleNamespace
        p2 = SimpleNamespace(dim=p.dim + 8, dim_interm=p.dim_interm, up=p.up, out_bias=p.out_bias)
        self.net1 = Interp(p2, n_classes)

        # self.net1 = nn.Sequential(
            
        #     nn.Conv3d(p.dim+8, n_classes, kernel_size=(1,3,3), padding=(0,1,1))
        # )             

        #self.skip = nn.Conv3d(p.dim + 8, n_classes, 1)

        # self.netX = nn.Conv3d(p.dim, n_classes, kernel_size=1)           
        from deap import utilities as ut
        print(f'decoder params (dim2={dim2}): {ut.count_parameters(self)/1e6:.3f}M')       

    def forward(self, feats):

        bs, _, n_frames = feats.shape[:3]

        feats = self.gn(feats)
        feats = self.proj_feats(feats)
        feats = self.conv(feats)

        feats = nn.functional.interpolate(feats, (n_frames, *self.base_size), mode='trilinear')

        feats = feats.permute(0, 2, 3, 4, 1)  # move channels to last dim

        feats_pe = self.feats_pe[None].repeat(bs, 1, 1, 1, 1)

        feats = torch.cat([feats, feats_pe], dim=4)

        feats = feats.flatten(1,3)

        if len(self.self_att.layers) > 0:
            feats = feats + self.self_att(feats)
            
        x = feats.view(bs, n_frames, *self.base_size, self.dim + 8).permute(0,4,1,2,3)

        x = self.gn2(x)
        #skip_x = self.skip(x)

        x = self.net1(x)
    
        #x = x + nn.functional.interpolate(skip_x, x.shape[2:], mode='trilinear')

        # x = self.netX(x)
        return x



DECODER_CLASSES.update({
    'none': lambda *x: None,
    'C': partial(ConvDecoder, depth=0),
    'C-D1': partial(ConvDecoder, depth=1),
})


class SelfAttReadouts(nn.Module):

    def __init__(self, backbone_name,  base_size, decoder, up, outputs, inp_img_size=224, n_frames = 1, dim=128, 
                conv_skip=None, layers=None,
                vdim_fac=None, low_rank_v=None,  low_rank_k=None, feats_pe=True,
                 n_heads=8, pe_dim=8, out_bias=None, shuffle_feats=False, timm_mode=False,
                 no_freeze=False, dim_interm=None, no_early_device_copy=False,
                   **unused_arguments):
        
        super().__init__()

        print('shuffle feats', shuffle_feats)
        print('decoder', decoder)

        # from deap.logger import AttDict

        # if isinstance(backbone_name, AttDict):
        #     backbone = backbone_name.init()
        if isinstance(backbone_name, partial):
            backbone = backbone_name()
        elif not isinstance(backbone_name, str):
            backbone = backbone_name
        else:
            if timm_mode:
                from deap.models.backbones import TimmAutoBackbone
                backbone = TimmAutoBackbone(backbone_name)
            else:
                from deap.models import backbones
                backbone = backbones.backbone_dict[backbone_name]()
        
        if not no_freeze:
            backbone.freeze()
            backbone.eval()

        self.backbone = backbone
        self.n_frames = n_frames
        self.dim = dim
        self.layers = layers
        
        self.feats_pe = feats_pe
        self.low_rank_v = low_rank_v
        self.low_rank_k = low_rank_k
        self.vdim_fac = vdim_fac
        self.dim_interm = dim_interm # for interp
        self.pe_dim = pe_dim
        self.up = up
        self.n_heads = n_heads
        self.no_early_device_copy = no_early_device_copy
        self.inp_dim = self.backbone.feature_dim()
        self.f_base_size = self.backbone.base_size(inp_img_size)
        self.inp_img_size = (inp_img_size, inp_img_size) if isinstance(inp_img_size, int) else inp_img_size
        self.base_size = (base_size, base_size) if isinstance(base_size, int) else base_size
        self.out_bias = out_bias

        # for shuffling
        if shuffle_feats:
            self.index_order = nn.Parameter(torch.multinomial(torch.ones(self.inp_dim), self.inp_dim), requires_grad=False)
        else:
            self.index_order = None

        if conv_skip is not None:

            self.conv_skip = nn.Sequential(
                nn.Conv3d(3, conv_skip, kernel_size=5, padding=2), nn.ReLU(), 
                nn.Conv3d(conv_skip, conv_skip, kernel_size=5, padding=2)
            )

            # hack for outputs
            self.conv2 = nn.ModuleDict({name: nn.Sequential(
                nn.Conv3d(2*conv_skip, conv_skip, kernel_size=5, padding=2), nn.ReLU(), 
                nn.Conv3d(conv_skip, n_outs, kernel_size=5, padding=2)
            ) for name, n_outs in outputs})

            outputs = [(o[0], conv_skip) for o in outputs]


        else:
            self.conv_skip = None

        if isinstance(decoder, str):
            decoder_cls = DECODER_CLASSES[decoder]
        else:
            decoder_cls = decoder
        print(self.inp_img_size)

        self.precomputed_features = None

        self.decoders = nn.ModuleDict()
        for name, n_outs in outputs:
            print('init', name, ', outputs:', n_outs)
            self.decoders[name] = decoder_cls(self, n_outs)
            

    def precompute_features(self, loader, device='cpu', dtype=None):
        if self.precomputed_features is None:
            self.precomputed_features = dict()
        device = next(self.parameters()).device
        import tqdm
        for sample in tqdm.tqdm(loader):
            with torch.no_grad():
                feats = self.compute_feats(sample, device=device)
            assert 'id' not in sample or isinstance(sample['id'][0], str)
            for f, id in zip(feats, sample['id']):
                if dtype is not None:
                    f = f.to(dtype)
                self.precomputed_features[id] = f.to(device)


    def compute_feats(self, sample, frame_i=None, device='cpu'):
        
        assert isinstance(sample, dict) and 'image' in sample
        
        x = sample['image']
        if not self.no_early_device_copy:
            x = x.to(device)
       
        if x.shape[3:5] != tuple(self.inp_img_size):
            # print(f'interpolate {x.shape[3:5]} to {self.inp_img_size}')
            x = torch.nn.functional.interpolate(x, (1, *self.inp_img_size)) # , mode='bicubic', antialias=True)

        # print(x.requires_grad)
        # print()

        if frame_i is None:
            feats = self.backbone(x)
        else:
            assert self.n_frames == 1
            feats = self.backbone(x[:, :, frame_i:frame_i+1])
        
        if self.index_order is not None:
            feats = feats[:, self.index_order]

        return feats

    def pack_feats(self, sample, device):
        if isinstance(self.precomputed_features[sample['id'][0]], dict):
            keys = list(self.precomputed_features[sample['id'][0]].keys())
            return {k: torch.stack([self.precomputed_features[s][k] for s in sample['id']]).to(device) for k in keys}
        else:
            return torch.stack([self.precomputed_features[s] for s in sample['id']]).to(device)


    def forward(self, sample):

        device = next(self.parameters()).device

        if self.precomputed_features is not None:
            # feats = torch.stack([self.precomputed_features[s] for s in sample['id']]).to(device)
            feats = self.pack_feats(sample, device)
        else:
            feats = self.compute_feats(sample, device=device)

        if self.conv_skip is not None:
            conv_out = self.conv_skip(sample['image'].to(device))
            fac = torch.prod(torch.tensor(self.up))
            conv_out = torch.nn.functional.interpolate(conv_out, (1, fac*self.base_size[0], fac*self.base_size[1]), mode='trilinear')

        out = dict(feats=feats)
        for name in self.decoders:
            out[name] = self.decoders[name](feats)
        
        # import ipdb; ipdb.set_trace()
        if self.conv_skip is not None:
            for name in self.conv2:
                out[name] = self.conv2[name](torch.cat([out[name], conv_out], dim=1))
                
        return out

    def get_name(self):
        return f'{self.__class__.__name__}-{self.dim}-{self.backbone.__class__.__name__}'






class SelfAttReadoutsFeatup(nn.Module):

    def __init__(self, version='dinov2', inp_img_size=224, target_size=None, outputs=(('out', 1),), mode='lin', 
                 base_size=None, up=None, out_bias=None):
        super().__init__()
        self.upsampler = torch.hub.load("mhamilton723/FeatUp", version, use_norm=True)

        for p in self.upsampler.parameters():
            p.requires_grad = False

        self.target_size = inp_img_size if target_size is None else target_size
        self.inp_img_size = (inp_img_size, inp_img_size) if isinstance(inp_img_size, int) else inp_img_size
        # self.guide_res = guide_res
        self.version = version
        dim = {'dinov2': 384}[version]

        from featup.util import norm
        self.norm = norm

        self.decoders = nn.ModuleDict()
        for name, n_classes in outputs:
            if mode == 'lin':
                self.decoders[name] = nn.Conv2d(dim, n_classes, kernel_size=1)
            elif mode == 'cnn':
                self.decoders[name] = nn.Sequential(
                    nn.Conv2d(dim, 32, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(32, n_classes, kernel_size=3, padding=1)
                )

    def forward(self, sample):
        device = next(self.parameters()).device

        x = sample['image'].to(device)

        if x.shape[3:5] != tuple(self.inp_img_size):
            x = torch.nn.functional.interpolate(x, (1, *self.inp_img_size))

        x = self.norm(x[:,:,0])
        # x_guide = torch.nn.functional.interpolate(x, self.guide_res)

    
        with torch.no_grad():
            lr_feats = self.upsampler.model(x)
            hr_feats = self.upsampler.upsampler(lr_feats, x)


        outputs = dict()
        for key, dec in self.decoders.items():
            pred = dec(hr_feats)
            pred = torch.nn.functional.interpolate(pred, self.target_size)
            outputs[key] = pred[:,:,None]

        return outputs

    def get_name(self):
        return f'{self.__class__.__name__}-{self.version}'






