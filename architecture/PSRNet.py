import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
import numbers

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')



class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class CrossPolarAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()

        self.dim = dim

        self.heads = heads
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

        self.mask_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),  
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )

    def forward(self, x_in, mask):
        """
        x_in: [B*P, C, H, W]
        mask: [B*P, C, H, W]
        """
        
        bp, c, h, w = x_in.shape
        p_num = 4
        b_num = bp//p_num

        x_in = x_in.view(b_num, p_num, c, h, w)
        x_in = rearrange(x_in, 'b p c h w -> (b h w) p c')  # [BHW, P, C]

        q = self.to_q(x_in)
        k = self.to_k(x_in)
        v = self.to_v(x_in)

        
        q = rearrange(q, 'b p (h c) -> b h p c', h=self.heads)
        k = rearrange(k, 'b p (h c) -> b h p c', h=self.heads)
        v = rearrange(v, 'b p (h c) -> b h p c', h=self.heads) 

        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        


        atten = (q @ k.transpose(-2, -1)) * self.rescale 

        mask = mask.view(b_num, p_num, c, h, w)

        mask_guidance = self.mask_proj(mask.view(-1, c, h, w))  
        mask_guidance = mask_guidance.view(b_num, p_num, c, h, w).mean(dim=2)  


        mask_bias = rearrange(mask_guidance, 'b p h w -> (b h w) p')  
        mask_bias = mask_bias.unsqueeze(1) - mask_bias.unsqueeze(2)   

        atten = atten + mask_bias.unsqueeze(1) 



        atten = atten.softmax(dim=-1)

        out = (atten @ v)

        out = rearrange(out, 'b h p c -> b p (h c)')



        out = self.proj(out)

        out = rearrange(out, '(b h w) p c -> b p c h w', b=b_num, h=h, w=w)

        out = out.view(b_num*p_num, c, h, w)

        return out



class Spectral_Atten(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.num_heads = heads
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)


        self.mask_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),  
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        )
    def forward(self, x_in, mask):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, c, h, w = x_in.shape
        q_in = self.q_dwconv(self.to_q(x_in))
        k_in = self.k_dwconv(self.to_k(x_in))
        v_in = self.v_dwconv(self.to_v(x_in))


        q = rearrange(q_in, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k_in, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v_in, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
  


        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        atten = (q @ k.transpose(-2, -1)) * self.rescale

      
        mask_bias = self.mask_proj(mask)  
        mask_bias = mask_bias.mean(dim=(2, 3)) 

        mask_bias = rearrange(mask_bias, 'b (head c)-> b head c', head=self.num_heads)


        mask_bias = mask_bias.unsqueeze(2) - mask_bias.unsqueeze(3)  

        atten = atten + mask_bias  
        
        atten = atten.softmax(dim=-1) 

        out = (atten @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.proj(out)


        return out




class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class PreNorm(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )
        

    def forward(self, x):
        return self.ffn(x)
    

class PreNorm_Pols(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv3d(dim, dim * mult, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False, groups=dim),
            GELU(),
            nn.Conv3d(dim * mult, dim, kernel_size=(1, 1, 1), padding=0, bias=False),
        )
        

    def forward(self, x):

        bp, c, h, w = x.shape
        p_num = 4
        b_num = bp//4

        x = x.view(b_num, p_num, c, h, w)

        x = x.permute(0, 2, 1, 3, 4) 

        out = self.ffn(x)
        out = out.permute(0, 2, 1, 3, 4)  
        out = out.view(b_num*p_num, c, h, w)



        return out



class SAM_Spectral(nn.Module):
    def __init__(self, dim, heads, num_blocks):
        super().__init__()

        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                LayerNorm(dim),
                Spectral_Atten(dim=dim, heads=heads),
                LayerNorm(dim),
                PreNorm(dim, mult=4)
            ]))
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        for (norm1, atten, norm2, ffn) in self.blocks:
            x = atten(norm1(x)) + x
            x = ffn(norm2(x)) + x
        return x
    
class SAM_SpectralPols(nn.Module):
    def __init__(self, dim, heads, num_blocks):
        super().__init__()

        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                LayerNorm(dim),
                CrossPolarAttention(dim=dim, heads=heads),
                LayerNorm(dim),
                PreNorm_Pols(dim, mult=4)
            ]))
    def forward(self, x, mask):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        for (norm1, atten, norm2, ffn) in self.blocks:
            x = atten(norm1(x), mask) + x
            x = ffn(norm2(x)) + x
        return x


class SAM_Spectral(nn.Module):
    def __init__(self, dim, heads, num_blocks):
        super().__init__()

        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                LayerNorm(dim),
                Spectral_Atten(dim=dim, heads=heads),
                LayerNorm(dim),
                PreNorm(dim, mult=4)
            ]))
    def forward(self, x, mask):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        for (norm1, atten, norm2, ffn) in self.blocks:
            x = atten(norm1(x), mask) + x
            x = ffn(norm2(x)) + x
        return x
    




class PSAM(nn.Module):
    def __init__(self, dim, heads, num_blocks):
        super().__init__()

        self.sam_spectral = SAM_Spectral(dim, heads, num_blocks)
        self.sam_specpols = SAM_SpectralPols(dim, heads, num_blocks)

        self.attention = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x, mask, type):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """

        if type=='spec':
            out = self.sam_spectral(x, mask)

        elif type=='pols':

            x_spec = self.sam_spectral(x, mask)
            x_pol = self.sam_specpols(x, mask)
            fusion_input = torch.cat([x_spec, x_pol], dim=1) 
            attn = self.attention(fusion_input) 

            out = attn * x_spec + (1 - attn) * x_pol

        else:
            print('The type of input data is error.')

        return out





class PSRNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=61, dim=32, deep_stage=3, num_blocks=[1, 1, 1], num_heads=[1, 2, 4]):
        super(PSRNet, self).__init__()
        self.dim = dim
        self.out_channels = out_channels
        self.stage = deep_stage
        self.SR = 2

        self.embedding1 = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=False)
        self.embedding2 = nn.Conv2d(61, dim, kernel_size=3, padding=1, bias=False)
        self.embedding = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False)
        
        self.down_sample = nn.Conv2d(dim, dim, 4, 2, 1, bias=False)
        self.up_sample = nn.ConvTranspose2d(dim, dim, stride=2, kernel_size=2, padding=0, output_padding=0)

        self.mapping = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=False)


        self.up_sample_out = nn.ConvTranspose2d(dim, dim, stride=2, kernel_size=2, padding=0, output_padding=0)




        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(deep_stage):
            self.encoder_layers.append(nn.ModuleList([
                PSAM(dim=dim_stage, heads=num_heads[i], num_blocks=num_blocks[i]),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2


        self.bottleneck = PSAM(dim=dim_stage, heads=num_heads[-1], num_blocks=num_blocks[-1])

        self.decoder_layers = nn.ModuleList([])
        for i in range(deep_stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                PSAM(dim=dim_stage // 2, heads=num_heads[deep_stage - 1 - i], num_blocks=num_blocks[deep_stage - 1 - i]),
            ]))
            dim_stage //= 2


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask, type='None'):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        if type=='spec':
            x = self.embedding1(x)
            mask = self.embedding2(mask)
            x = torch.cat((x, mask), dim=1)
            fea = self.embedding(x)
        elif type=='pols':
            b, p, c, h, w = x.shape
 
            c_out = mask.shape[2]
            mask = mask.view(b*p, c_out, h, w)
 
            x = x.view(b*p, c, h, w)

            x = self.embedding1(x)
            mask = self.embedding2(mask)
            x = torch.cat((x, mask), dim=1)
            fea = self.embedding(x)
        else:
            print('The type of input data is error.')

        residual = fea

        fea_encoder = []
        masks = []
        for (Attention, FeaDownSample, MASKDown) in self.encoder_layers:
            fea = Attention(fea, mask, type)
            masks.append(mask)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            mask = MASKDown(mask)

        fea = self.bottleneck(fea, mask, type)
 
        for i, (FeaUpSample, Fution, Attention) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            mask = masks[self.stage - 1 - i]
            fea = Attention(fea, mask, type)
 

        if type=='spec':
            out = fea + residual
            out = self.up_sample_out(out)

            out = self.mapping(out)
        elif type=='pols':
            out = fea + residual
            out = self.up_sample_out(out)
            out = self.mapping(out)

            out = out.view(b, p, c_out, h*self.SR, w*self.SR)
        else:
            print('The type of input data is error.')



        return out












