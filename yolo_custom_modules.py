"""
Custom YOLO modules for YOLOv12
Add this file to your project to support A2C2f and other custom modules
"""
import torch
import torch.nn as nn

try:
    from ultralytics.nn.modules.block import C2f
    from ultralytics.nn.modules.conv import Conv, autopad
except ImportError:
    print("Warning: Could not import from ultralytics. Using fallback definitions.")
    
    class Conv(nn.Module):
        """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
        default_act = nn.SiLU()
        
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
            super().__init__()
            self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        
        def forward(self, x):
            return self.act(self.bn(self.conv(x)))
    
    def autopad(k, p=None, d=1):
        """Pad to 'same' shape outputs."""
        if d > 1:
            k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p
    
    class C2f(nn.Module):
        """Faster Implementation of CSP Bottleneck with 2 convolutions."""
        
        def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
            super().__init__()
            self.c = int(c2 * e)
            self.cv1 = Conv(c1, 2 * self.c, 1, 1)
            self.cv2 = Conv((2 + n) * self.c, c2, 1)
            self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        
        def forward(self, x):
            y = list(self.cv1(x).chunk(2, 1))
            y.extend(m(y[-1]) for m in self.m)
            return self.cv2(torch.cat(y, 1))
    
    class Bottleneck(nn.Module):
        """Standard bottleneck."""
        
        def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
            super().__init__()
            c_ = int(c2 * e)
            self.cv1 = Conv(c1, c_, k[0], 1)
            self.cv2 = Conv(c_, c2, k[1], 1, g=g)
            self.add = shortcut and c1 == c2
        
        def forward(self, x):
            return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Attention(nn.Module):
    """Attention module for A2C2f."""
    
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = max(dim // num_heads, 32)
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)
    
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        qkv = self.qkv(x)
        q, k, v = qkv.split([self.key_dim * self.num_heads, 
                             self.key_dim * self.num_heads, 
                             C], dim=1)
        
        q = q.reshape(B, self.num_heads, self.key_dim, N).permute(0, 1, 3, 2)
        k = k.reshape(B, self.num_heads, self.key_dim, N)
        v = v.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        
        x_out = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        x_out = self.proj(x_out)
        
        return x_out + self.pe(v.reshape(B, C, H, W))


class A2C2f(nn.Module):
    """
    Attention-enhanced C2f module (A2C2f) for YOLOv12.
    Combines CSP bottleneck structure with attention mechanism.
    """
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        # Bottleneck blocks
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) 
            for _ in range(n)
        )
        
        # Attention module
        self.attn = Attention(self.c, num_heads=4, attn_ratio=0.5)
    
    def forward(self, x):
        # Split and process
        y = list(self.cv1(x).chunk(2, 1))
        
        # Apply bottleneck blocks with attention
        for i, m in enumerate(self.m):
            y_m = m(y[-1])
            # Apply attention to every other block
            if i % 2 == 0:
                y_m = self.attn(y_m)
            y.append(y_m)
        
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""
    
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# Register the custom module with ultralytics
def register_custom_modules():
    """Register custom modules so they can be loaded from checkpoints."""
    try:
        import ultralytics.nn.modules.block as block_module
        
        # Add custom modules to the block module
        if not hasattr(block_module, 'A2C2f'):
            block_module.A2C2f = A2C2f
            print("✅ Registered A2C2f module")
        
        if not hasattr(block_module, 'Attention'):
            block_module.Attention = Attention
            print("✅ Registered Attention module")
            
    except Exception as e:
        print(f"⚠️ Could not register custom modules: {e}")
        print("   You may need to update ultralytics or use a different approach")


# Auto-register when module is imported
register_custom_modules()