import torch
import torch.nn as nn
import torch.nn.functional as F


# Basic Conv: Conv2d + BN + SiLU
class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# Bottleneck used inside C3-like blocks
class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, shortcut=True, expansion=0.5):
        super().__init__()
        hidden = int(out_ch * expansion)
        self.cv1 = Conv(in_ch, hidden, k=3)  # Changed to k=3 like official
        self.cv2 = Conv(hidden, out_ch, k=3)
        self.add = shortcut and in_ch == out_ch

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        if self.add:
            return x + y
        return y


# C3k block with k bottlenecks (nested inside C3k2 when shortcut=True)
class C3k(nn.Module):
    def __init__(self, in_ch, out_ch, n=2, shortcut=True, expansion=0.5):
        super().__init__()
        hidden = int(out_ch * expansion)
        self.cv1 = Conv(in_ch, hidden, k=1)
        self.cv2 = Conv(in_ch, hidden, k=1)
        self.cv3 = Conv(2 * hidden, out_ch, k=1)
        self.m = nn.Sequential(*[Bottleneck(hidden, hidden, shortcut, expansion=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


# C3k2: Uses Bottleneck for simple layers, C3k for complex layers with shortcut=True
class C3k2(nn.Module):
    def __init__(self, in_ch, out_ch, n=1, shortcut=False, expansion=0.5):
        super().__init__()
        hidden = int(out_ch * expansion)
        self.cv1 = Conv(in_ch, hidden, k=1)
        self.cv2 = Conv(in_ch, hidden, k=1)
        
        # Use C3k for shortcut=True (layers 6, 8, 22), Bottleneck otherwise
        if shortcut and n > 0:
            self.m = nn.ModuleList([C3k(hidden, hidden, n=2, shortcut=True, expansion=1.0)])
        else:
            self.m = nn.ModuleList([Bottleneck(hidden, hidden, shortcut=False, expansion=1.0) for _ in range(n)])
        
        # Final conv expects 2*hidden from cv1 pathway + cv2 pathway
        self.cv3 = Conv(2 * hidden, out_ch, k=1)

    def forward(self, x):
        # Two pathways like official implementation
        y1 = self.cv1(x)
        
        # Process through bottlenecks/C3k
        for module in self.m:
            y1 = module(y1)
        
        # Second pathway (no processing)
        y2 = self.cv2(x)
        
        # Concatenate and merge
        return self.cv3(torch.cat((y1, y2), dim=1))


# SPPF: spatial pyramid pooling - fast
class SPPF(nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.cv1 = Conv(in_ch, out_ch // 2, k=1)
        self.cv2 = Conv(out_ch // 2 * 4, out_ch, k=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        return self.cv2(torch.cat((x, y1, self.m(y1), self.m(self.m(y1))), dim=1))


# Depthwise Convolution
class DWConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=k//2, groups=in_ch, bias=False)
        self.bn = nn.BatchNorm2d(in_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# Attention module for C2PSA
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # QKV projection
        self.qkv = Conv(dim, dim * 2, k=1)  # Changed to 2x for Q,K,V
        self.proj = Conv(dim, dim, k=1)
        
        # Position encoding
        self.pe = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Apply position encoding
        pe = self.pe(x)
        
        # QKV projection
        qkv = self.qkv(x)
        q, k = qkv.chunk(2, dim=1)
        v = x + pe
        
        # Reshape for multi-head attention
        q = q.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        
        # Attention
        attn = (q @ k) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        return self.proj(out)


# PSA Block
class PSABlock(nn.Module):
    def __init__(self, dim, num_heads=8, expansion=2):
        super().__init__()
        self.attn = Attention(dim, num_heads)
        
        # Feed-forward network
        hidden = dim * expansion
        self.ffn = nn.Sequential(
            Conv(dim, hidden, k=1),
            Conv(hidden, dim, k=1)
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


# C2PSA - C2 block with Position-aware Self-Attention
class C2PSA(nn.Module):
    def __init__(self, in_ch, out_ch, n=1, expansion=0.5, num_heads=8):
        super().__init__()
        hidden = int(in_ch * expansion)  # Use expansion to reduce channels
        self.cv1 = Conv(in_ch, hidden, k=1)
        self.cv2 = Conv(in_ch, hidden, k=1)
        
        # PSA blocks on reduced channels
        self.m = nn.Sequential(*[PSABlock(hidden, num_heads=num_heads) for _ in range(n)])
        
        # Final output projection (not in official, but needed for proper output)
        self.cv3 = Conv(hidden * 2, out_ch, k=1) if hidden * 2 != out_ch else nn.Identity()

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        
        # Process y1 through PSA
        y1 = self.m(y1)
        
        # Concatenate pathways
        out = torch.cat((y1, y2), dim=1)
        
        # Project to output channels if needed
        if isinstance(self.cv3, nn.Identity):
            return out
        return self.cv3(out)


# Simple Concat wrapper to match the listing behavior
class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.d = dim

    def forward(self, xs):
        return torch.cat(xs, dim=self.d)


# DFL - Distribution Focal Loss layer for bbox refinement
class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.c1 = c1
        # Weight for converting distribution to single value
        x = torch.arange(c1, dtype=torch.float)
        self.register_buffer('weight', x.view(1, c1, 1, 1))

    def forward(self, x):
        # x shape: [B, 4*c1, H, W]
        B, _, H, W = x.shape
        # Reshape to [B, 4, c1, H, W]
        x = x.view(B, 4, self.c1, H, W)
        # Apply softmax over c1 dimension
        x = F.softmax(x, dim=2)
        # Weighted sum to get single value per bbox coordinate
        x = (x * self.weight).sum(dim=2)
        return x


# Proto - Prototype mask generation network
class Proto(nn.Module):
    def __init__(self, in_ch, num_protos=32):
        super().__init__()
        self.cv1 = Conv(in_ch, in_ch, k=3)
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.cv2 = Conv(in_ch, in_ch, k=3)
        self.cv3 = Conv(in_ch, num_protos, k=1)

    def forward(self, x):
        x = self.cv1(x)
        x = self.upsample(x)
        x = self.cv2(x)
        return self.cv3(x)


# Segment head: produces detection + segmentation outputs (ultralytics style)
class SegmentHead(nn.Module):
    def __init__(self, num_classes=15, num_masks=32, num_protos=32, in_channels=(128, 256, 512)):
        super().__init__()
        c1, c2, c3 = in_channels
        self.num_classes = num_classes
        self.num_masks = num_masks
        self.nc = num_classes
        self.reg_max = 16  # DFL channels
        
        # Bounding box regression heads (cv2) - outputs 4*reg_max for DFL
        self.cv2 = nn.ModuleList([
            nn.Sequential(
                Conv(c1, 64, k=3),
                Conv(64, 64, k=3),
                nn.Conv2d(64, 4 * self.reg_max, 1)
            ),
            nn.Sequential(
                Conv(c2, 64, k=3),
                Conv(64, 64, k=3),
                nn.Conv2d(64, 4 * self.reg_max, 1)
            ),
            nn.Sequential(
                Conv(c3, 64, k=3),
                Conv(64, 64, k=3),
                nn.Conv2d(64, 4 * self.reg_max, 1)
            )
        ])
        
        # Classification heads (cv3) - uses DWConv
        self.cv3 = nn.ModuleList([
            nn.Sequential(
                nn.Sequential(DWConv(c1, c1, k=3), Conv(c1, c1, k=1)),
                nn.Sequential(DWConv(c1, c1, k=3), Conv(c1, c1, k=1)),
                nn.Conv2d(c1, num_classes, 1)
            ),
            nn.Sequential(
                nn.Sequential(DWConv(c2, c2, k=3), Conv(c2, c1, k=1)),
                nn.Sequential(DWConv(c1, c1, k=3), Conv(c1, c1, k=1)),
                nn.Conv2d(c1, num_classes, 1)
            ),
            nn.Sequential(
                nn.Sequential(DWConv(c3, c3, k=3), Conv(c3, c1, k=1)),
                nn.Sequential(DWConv(c1, c1, k=3), Conv(c1, c1, k=1)),
                nn.Conv2d(c1, num_classes, 1)
            )
        ])
        
        # Mask coefficient heads (cv4)
        self.cv4 = nn.ModuleList([
            nn.Sequential(
                Conv(c1, num_masks, k=3),
                Conv(num_masks, num_masks, k=3),
                nn.Conv2d(num_masks, num_masks, 1)
            ),
            nn.Sequential(
                Conv(c2, num_masks, k=3),
                Conv(num_masks, num_masks, k=3),
                nn.Conv2d(num_masks, num_masks, 1)
            ),
            nn.Sequential(
                Conv(c3, num_masks, k=3),
                Conv(num_masks, num_masks, k=3),
                nn.Conv2d(num_masks, num_masks, 1)
            )
        ])
        
        # DFL layer
        self.dfl = DFL(self.reg_max)
        
        # Proto generation network
        self.proto = Proto(c1, num_protos)

    def forward(self, feats):
        # feats = [small (P3), medium (P4), large (P5)]
        f1, f2, f3 = feats
        
        # Bbox regression (with DFL)
        bbox1 = self.cv2[0](f1)
        bbox2 = self.cv2[1](f2)
        bbox3 = self.cv2[2](f3)
        
        # Apply DFL to get refined bboxes
        bbox1_dfl = self.dfl(bbox1)
        bbox2_dfl = self.dfl(bbox2)
        bbox3_dfl = self.dfl(bbox3)
        
        # Classification
        cls1 = self.cv3[0](f1)
        cls2 = self.cv3[1](f2)
        cls3 = self.cv3[2](f3)
        
        # Combine bbox + cls for detection output
        det1 = torch.cat([bbox1_dfl, cls1], dim=1)
        det2 = torch.cat([bbox2_dfl, cls2], dim=1)
        det3 = torch.cat([bbox3_dfl, cls3], dim=1)
        
        # Mask coefficients
        mask1 = self.cv4[0](f1)
        mask2 = self.cv4[1](f2)
        mask3 = self.cv4[2](f3)
        
        # Prototype masks
        protos = self.proto(f1)
        
        return {
            'detections': [det1, det2, det3],  # bbox(4) + classes
            'masks': [mask1, mask2, mask3],
            'protos': protos
        }


# Full model wired per the supplied layer list (approximation of exact ultralytics graph)
class YOLOCustom(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        # We'll follow the layer list and build modules in order
        self.layers = nn.ModuleList()

        # 0 Conv(3,32,3,2)
        self.layers.append(Conv(3, 32, k=3, s=2))
        # 1 Conv(32,64,3,2)
        self.layers.append(Conv(32, 64, k=3, s=2))
        # 2 C3k2(64,128,1,False,0.25) -> n=1, shortcut=False, expansion=0.25
        self.layers.append(C3k2(64, 128, n=1, shortcut=False, expansion=0.25))
        # 3 Conv(128,128,3,2)
        self.layers.append(Conv(128, 128, k=3, s=2))
        # 4 C3k2(128,256,1,False,0.25)
        self.layers.append(C3k2(128, 256, n=1, shortcut=False, expansion=0.25))
        # 5 Conv(256,256,3,2)
        self.layers.append(Conv(256, 256, k=3, s=2))
        # 6 C3k2(256,256,1,True) -> n=1, shortcut=True, expansion=0.5 (default when not specified)
        self.layers.append(C3k2(256, 256, n=1, shortcut=True, expansion=0.5))
        # 7 Conv(256,512,3,2)
        self.layers.append(Conv(256, 512, k=3, s=2))
        # 8 C3k2(512,512,1,True) -> expansion=0.5
        self.layers.append(C3k2(512, 512, n=1, shortcut=True, expansion=0.5))
        # 9 SPPF(512,512,5)
        self.layers.append(SPPF(512, 512, k=5))
        # 10 C2PSA(512,512,1)
        self.layers.append(C2PSA(512, 512, n=1))
        # 11 Upsample
        self.layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        # 12 Concat
        self.layers.append(Concat(1))
        # 13 C3k2(768,256,1,False) -> expansion defaults to 0.5
        self.layers.append(C3k2(768, 256, n=1, shortcut=False, expansion=0.5))
        # 14 Upsample
        self.layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        # 15 Concat
        self.layers.append(Concat(1))
        # 16 C3k2(512,128,1,False)
        self.layers.append(C3k2(512, 128, n=1, shortcut=False, expansion=0.5))
        # 17 Conv(128,128,3,2)
        self.layers.append(Conv(128, 128, k=3, s=2))
        # 18 Concat
        self.layers.append(Concat(1))
        # 19 C3k2(384,256,1,False)
        self.layers.append(C3k2(384, 256, n=1, shortcut=False, expansion=0.5))
        # 20 Conv(256,256,3,2)
        self.layers.append(Conv(256, 256, k=3, s=2))
        # 21 Concat
        self.layers.append(Concat(1))
        # 22 C3k2(768,512,1,True)
        self.layers.append(C3k2(768, 512, n=1, shortcut=True, expansion=0.5))
        # 23 Segment head [15,32,32,[128,256,512]] -> num_classes=15, num_masks=32, num_protos=32
        self.seg = SegmentHead(num_classes=num_classes, num_masks=32, num_protos=32, in_channels=(128, 256, 512))

    def forward(self, x):
        # We'll keep a list of outputs and emulate the indexing used in the listing
        outputs = []

        # 0
        x0 = self.layers[0](x)
        outputs.append(x0)  # idx 0
        # 1
        x1 = self.layers[1](x0)
        outputs.append(x1)  # 1
        # 2
        x2 = self.layers[2](x1)
        outputs.append(x2)  # 2
        # 3
        x3 = self.layers[3](x2)
        outputs.append(x3)  # 3
        # 4
        x4 = self.layers[4](x3)
        outputs.append(x4)  # 4
        # 5
        x5 = self.layers[5](x4)
        outputs.append(x5)  # 5
        # 6
        x6 = self.layers[6](x5)
        outputs.append(x6)  # 6
        # 7
        x7 = self.layers[7](x6)
        outputs.append(x7)  # 7
        # 8
        x8 = self.layers[8](x7)
        outputs.append(x8)  # 8
        # 9
        x9 = self.layers[9](x8)
        outputs.append(x9)  # 9
        # 10
        x10 = self.layers[10](x9)
        outputs.append(x10)  # 10
        # 11 upsample x10
        x11 = self.layers[11](x10)
        outputs.append(x11)  # 11
        # 12 concat [-1,6] -> [x11, x6]
        x12 = self.layers[12]([x11, outputs[6]])
        outputs.append(x12)  # 12
        # 13 C3k2 on concat
        x13 = self.layers[13](x12)
        outputs.append(x13)  # 13
        # 14 upsample x13
        x14 = self.layers[14](x13)
        outputs.append(x14)  # 14
        # 15 concat [-1,4] -> [x14, x4]
        x15 = self.layers[15]([x14, outputs[4]])
        outputs.append(x15)  # 15
        # 16 C3k2
        x16 = self.layers[16](x15)
        outputs.append(x16)  # 16
        # 17 Conv(128,128,3,2) on x16
        x17 = self.layers[17](x16)
        outputs.append(x17)  # 17
        # 18 concat [-1,13] -> [x17, x13]
        x18 = self.layers[18]([x17, outputs[13]])
        outputs.append(x18)  # 18
        # 19 C3k2
        x19 = self.layers[19](x18)
        outputs.append(x19)  # 19
        # 20 Conv(256,256,3,2)
        x20 = self.layers[20](x19)
        outputs.append(x20)  # 20
        # 21 concat [-1,10] -> [x20, x10]
        x21 = self.layers[21]([x20, outputs[10]])
        outputs.append(x21)  # 21
        # 22 C3k2
        x22 = self.layers[22](x21)
        outputs.append(x22)  # 22
        # 23 Segment head on [16,19,22] -> outputs indices 16,19,22
        seg_out = self.seg([outputs[16], outputs[19], outputs[22]])
        outputs.append(seg_out)

        return seg_out

    def get_layer_outputs(self, x):
        """Return all intermediate layer outputs for debugging"""
        outputs = []
        # Forward through all layers and collect outputs
        for i, layer in enumerate(self.layers[:11]):
            if i == 0:
                outputs.append(layer(x))
            else:
                outputs.append(layer(outputs[-1]))
        
        # Layer 11-22 with skip connections
        x11 = self.layers[11](outputs[10])
        outputs.append(x11)
        x12 = self.layers[12]([x11, outputs[6]])
        outputs.append(x12)
        # ... continue pattern
        return outputs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Comprehensive test with GPU support
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    
    model = YOLOCustom(num_classes=15).to(device)
    total_params = count_parameters(model)
    print(f'Model built. Total params: {total_params:,}')
    
    # Test forward pass
    x = torch.randn(1, 3, 640, 640).to(device)
    with torch.no_grad():
        out = model(x)
    
    print('\n=== Output Shapes ===')
    print(f"Detections P3: {out['detections'][0].shape}")  # [1, 4+15, H/8, W/8]
    print(f"Detections P4: {out['detections'][1].shape}")  # [1, 4+15, H/16, W/16]
    print(f"Detections P5: {out['detections'][2].shape}")  # [1, 4+15, H/32, W/32]
    print(f"Masks P3: {out['masks'][0].shape}")           # [1, 32, H/8, W/8]
    print(f"Masks P4: {out['masks'][1].shape}")           # [1, 32, H/16, W/16]
    print(f"Masks P5: {out['masks'][2].shape}")           # [1, 32, H/32, W/32]
    print(f"Protos: {out['protos'].shape}")               # [1, 32, H/2, W/2]
    
    # Layer parameter breakdown
    print('\n=== Parameter Distribution ===')
    for i, layer in enumerate(model.layers):
        params = sum(p.numel() for p in layer.parameters())
        if params > 0:
            print(f"Layer {i:2d}: {params:>10,} params - {layer.__class__.__name__}")
    
    seg_params = sum(p.numel() for p in model.seg.parameters())
    print(f"Segment Head: {seg_params:>10,} params")

