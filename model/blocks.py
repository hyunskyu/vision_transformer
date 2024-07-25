import torch
import torch.nn as nn
from itertools import repeat
import collections.abc


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0,):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])


    def forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    


class Attention(nn.Module):

    def __init__(self, dim, rope=None, num_heads=16, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x, xpos):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B,N,3, self.num_heads, C // self.num_heads).transpose(1,3)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        
        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        attn = (q@k.transpose(-2,-1))*self.scale
        attn = attn.softmax(dim=-1) # Query 토큰에 대해 Key 토큰을 softmax
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
    
        return x

class CrossAttention(nn.Module):

    def __init__(self, dim, rope=None, num_heads=16, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.projq = nn.Linear(dim,dim, bias=qkv_bias)
        self.projk = nn.Linear(dim,dim, bias=qkv_bias)
        self.projv = nn.Linear(dim,dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, query, key, value, xpos, ypos):
        B,N,C= query.shape

        q = self.projq(query).reshape(B,N,self.num_heads, C //self.num_heads)
        k = self.projk(key).reshape(B,N,self.num_heads, C //self.num_heads)
        v = self.projv(value).reshape(B,N,self.num_heads, C //self.num_heads)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        attn = (q@k.transpose(-2,-1))*self.scale
        attn = attn.softmax(dim=-1) # Query 토큰에 대해 Key 토큰을 softmax
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Encoder(nn.Module):

    def __init__(self,dim, num_heads, mlp_ratio=4, qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, rope=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path >0. else nn.Identity() # Sub block을 확률적으로 스킵
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, xpos):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Decoder(nn.Module):
    def __init__(self,dim, num_heads, mlp_ratio=4, qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=None):
        super().__init__()
        self.norm1= norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()
        self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path >0. else nn.Identity() # Sub block을 확률적으로 스킵
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x,y, xpos,ypos):
        x = x + self.drop_path(self.attn(self.norm1(x),xpos))
        y_ = self.norm_y(y)
        x = x + self.drop_path(self.cross_attn(self.norm2(x),y_,y_,xpos,ypos))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x,y