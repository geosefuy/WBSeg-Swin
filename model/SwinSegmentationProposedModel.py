import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SwinTransformer import PatchEmbedding, BasicLayer, PatchMerging

class SwinDecoder(nn.Module):
    def __init__(self, embed_dim=48):
        super().__init__()
        # Swin Transformer stages: 
        # stage1: embed_dim
        # stage2: embed_dim * 2
        # stage3: embed_dim * 4
        # stage4: embed_dim * 8

        ch1 = embed_dim
        ch2 = embed_dim * 2
        ch3 = embed_dim * 4
        ch4 = embed_dim * 8

        self.up4 = AttentionConverge(ch4 + ch3, ch3)  # fuse x4 with x3
        self.up3 = AttentionConverge(ch3 + ch2, ch2)  # fuse with x2
        self.up2 = AttentionConverge(ch2 + ch1, ch1)  # fuse with x1

    def forward(self, features):
        x1, x2, x3, x4 = features  # From encoder

        d4 = self.up4(x4, x3)  # fuse stage4 + stage3
        d3 = self.up3(d4, x2)  # fuse with stage2
        d2 = self.up2(d3, x1)  # fuse with stage1

        return d2  # (B, embed_dim, H/4, W/4)
    
class AttentionConverge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = torch.cat([x, skip], dim=1)
        return x
    
class SwinTransformerEncoder(nn.Module):
    def __init__(self, img_size=(1024,512), embed_dim=48,patch_size=4, in_chans=3, num_classes=1000,
                 depths=[2, 2, 6, 2], num_heads=[2, 4, 8, 16],
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
      
      
    def forward(self, x):
        x = self.patch_embed(x)  # B, H/4 * W/4, C
        x = self.pos_drop(x)

        # Store features from each stage
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Convert to 4D tensor for decoder compatibility
            if i < len(self.layers) - 1:  # Not the last layer
                H, W = layer.input_resolution
                if hasattr(layer, 'downsample') and layer.downsample is not None:
                    H, W = H // 2, W // 2  # After downsampling
                B, L, C = x.shape
                feature_4d = x.view(B, H, W, C).permute(0, 3, 1, 2)  # B, C, H, W
            else:  # Last layer
                H, W = layer.input_resolution
                B, L, C = x.shape
                feature_4d = x.view(B, H, W, C).permute(0, 3, 1, 2)  # B, C, H, W
            features.append(feature_4d)

        return features

class SwinSegmentationProposedModel(nn.Module):
    def __init__(self, img_size=(1024,512), num_classes=2, embed_dim=48, mlp_ratio=4., patch_size=4, window_size=8, depths=[2, 2, 6, 2]):
        super().__init__()
        self.img_size = img_size
        self.encoder = SwinTransformerEncoder(img_size=img_size, num_classes=num_classes, embed_dim=embed_dim, 
                                              mlp_ratio=mlp_ratio, patch_size=patch_size, window_size=window_size, depths=depths)
        self.decoder = SwinDecoder(embed_dim=embed_dim)
        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        x = self.decoder(features)
        x = self.head(x)
        x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
        return x