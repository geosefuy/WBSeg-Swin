import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.
    """
    def __init__(self, img_size=(1024,512), patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # B, embed_dim, H//patch_size, W//patch_size
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        x = self.norm(x)
        return x

class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with relative position bias.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Define relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        # Get pair-wise relative position indices
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
    
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)
        
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def window_partition(x, window_size):
    """
    Partition into non-overlapping windows.
    Args:
        x: (B, H, W, C)
        window_size: window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition.
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Identity()  # Simplified
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        
        self.register_buffer("attn_mask", attn_mask)
    
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class PatchMerging(nn.Module):
    """
    Patch Merging Layer - reduces spatial resolution and increases channels.
    """
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        
        x = x.view(B, H, W, C)
        
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x
    
class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        
        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                               num_heads=num_heads, window_size=window_size,
                               shift_size=0 if (i % 2 == 0) else window_size // 2,
                               mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                               drop=drop, attn_drop=attn_drop,
                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)])
        
        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None
    
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class SwinTransformer(nn.Module):
    """
    Swin Transformer
    """
    def __init__(self, img_size=(1024,512), patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=48, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=8, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        
        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                             input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                             patches_resolution[1] // (2 ** i_layer)),
                             depth=depths[i_layer],
                             num_heads=num_heads[i_layer],
                             window_size=window_size,
                             mlp_ratio=self.mlp_ratio,
                             qkv_bias=qkv_bias,
                             drop=drop_rate,
                             attn_drop=attn_drop_rate,
                             drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                             downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)
        
        self.norm = nn.LayerNorm(self.num_features)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x