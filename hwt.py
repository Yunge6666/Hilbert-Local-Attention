# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")

_gilbert_cache = {}

def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)

def generate2d(x: int, y: int, ax: int, ay: int, bx: int, by: int, result):
    w = abs(ax + ay)
    h = abs(bx + by)
    dax, day = sgn(ax), sgn(ay)
    dbx, dby = sgn(bx), sgn(by)

    if h == 1 or w == 1:
        if h == 1:
            for _ in range(w):
                result.append((x, y))
                x, y = x + dax, y + day
        elif w == 1:
            for _ in range(h):
                result.append((x, y))
                x, y = x + dbx, y + dby
        return

    ax2, ay2 = ax // 2, ay // 2
    bx2, by2 = bx // 2, by // 2
    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2 * w > 3 * h:
        if w2 % 2 and w > 2:
            ax2, ay2 = ax2 + dax, ay2 + day
        generate2d(x, y, ax2, ay2, bx, by, result)
        generate2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by, result)
    else:
        if h2 % 2 and h > 2:
            bx2, by2 = bx2 + dbx, by2 + dby
        generate2d(x, y, bx2, by2, ax2, ay2, result)
        generate2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2, result)
        generate2d(x + (ax - dax) + (bx2 - dbx),
                   y + (ay - day) + (by2 - dby),
                   -bx2, -by2, -(ax - ax2), -(ay - ay2), result)

def gilbert2d(width, height):
    result = []
    if width >= height:
        generate2d(0, 0, width, 0, 0, height, result)
    else:
        generate2d(0, 0, 0, height, width, 0, result)
    return result

class GilbertPathCache:
    
    def __init__(self):
        self.cache = {}
        
    def get_or_create_path(self, H, W):
        key = (H, W)
        if key not in self.cache:
            path = gilbert2d(W, H)
            
            forward_map = torch.zeros((H, W), dtype=torch.long)
            reverse_map = torch.zeros((H * W, 2), dtype=torch.long)
            
            for idx, (x, y) in enumerate(path[:H*W]):
                if y < H and x < W:
                    forward_map[y, x] = idx
                    reverse_map[idx, 0] = y
                    reverse_map[idx, 1] = x
            
            self.cache[key] = {
                'path': path,
                'forward_map': forward_map,
                'reverse_map': reverse_map,
                'H': H,
                'W': W
            }
        
        return self.cache[key]
    
    def precompute_paths(self, resolutions):
        for H, W in resolutions:
            self.get_or_create_path(H, W)
    
    def clear_cache(self):
        self.cache.clear()

_global_gilbert_cache = GilbertPathCache()
_rpb_index_cache = {}
def tensor_to_gilbert_path(x, cache=None):
    """ 
    Args:
        x: Input tensor, shape (B, H, W, C)
        cache: Optional GilbertPathCache instance, use global cache if None
    Returns:
        Reordered tensor, shape (B, H*W, C)
    """
    B, H, W, C = x.shape
    device = x.device
    
    # use cache
    if cache is None:
        cache = _global_gilbert_cache
    
    # get or create path mapping
    path_info = cache.get_or_create_path(H, W)
    reverse_map = path_info['reverse_map'].to(device)  # (H*W, 2)
    
 
    y_indices = reverse_map[:, 0]  # (H*W,)
    x_indices = reverse_map[:, 1]  # (H*W,)
    
    gilbert_tensor = x[:, y_indices, x_indices, :]  # (B, H*W, C)
    
    return gilbert_tensor

def gilbert_tensor_to_2d(x, H, W, cache=None):
    """
    Convert Gilbert sequence tensor back to 2D layout (using cache optimization)    
    Args:
        x: Gilbert sequence tensor, shape (B, H*W, C)
        H: Target height
        W: Target width
        cache: Optional GilbertPathCache instance, use global cache if None
    Returns:
        2D layout tensor, shape (B, H, W, C)
    """
    B, N, C = x.shape
    device = x.device
    
    # use cache
    if cache is None:
        cache = _global_gilbert_cache
    
    # get or create path mapping
    path_info = cache.get_or_create_path(H, W)
    reverse_map = path_info['reverse_map'].to(device)  # (H*W, 2)
    
    # create output tensor
    output_2d = torch.zeros((B, H, W, C), dtype=x.dtype, device=device)
    
    valid_n = min(N, H * W)
    if valid_n > 0:
        y_indices = reverse_map[:valid_n, 0]  # (valid_n,)
        x_indices = reverse_map[:valid_n, 1]  # (valid_n,)
        output_2d[:, y_indices, x_indices, :] = x[:, :valid_n, :]
    
    return output_2d

def gilbert_tensor_to_windows(x, window_size, reverse_map, shift_size=0):
    """
    Split Gilbert sequence tensor into windows by window_size
    Args:
        x: Gilbert sequence tensor, shape (B, N, C)
        window_size: window size
    Returns:
        Windowed tensor, shape (B*num_windows, window_size*window_size, C)
    """
    B, N, C = x.shape
    window_size_sq = window_size * window_size
    
    num_windows = N // window_size_sq
    if N % window_size_sq != 0:
        target_N = num_windows * window_size_sq
        if N > target_N:
            x = x[:, :target_N, :]
        else:
            padding = torch.zeros((B, target_N - N, C), dtype=x.dtype, device=x.device)
            x = torch.cat([x, padding], dim=1)
        N = target_N
        num_windows = N // window_size_sq
    
    # reshape to window format
    x_windows = x.contiguous().view(B, num_windows, window_size_sq, C)
    x_windows = x_windows.view(B * num_windows, window_size_sq, C)

    return x_windows

def gilbert_windows_to_tensor(x_windows, B, H, W):
    """
    Convert windowed tensor back to Gilbert sequence tensor
    Args:
        x_windows: Windowed tensor, shape (B*num_windows, window_size*window_size, C)
        B: Batch size
        H: Original height
        W: Original width
    Returns:
        Gilbert sequence tensor, shape (B, H*W, C)
    """
    BW, window_size_sq, C = x_windows.shape
    num_windows = BW // B
    
    # reshape back to Gilbert sequence tensor
    x = x_windows.view(B, num_windows, window_size_sq, C)
    x = x.view(B, num_windows * window_size_sq, C)
    
    target_size = H * W
    if x.shape[1] > target_size:
        x = x[:, :target_size, :]
    
    return x

def gilbert_shift_forward(x, shift_size):
    """
    Forward shift Gilbert sequence tensor
    Args:
        x: Gilbert sequence tensor, shape (B, N, C)
        shift_size: Shift size
    Returns:
        Shifted Gilbert sequence tensor, shape (B, N, C)
    """
    if shift_size > 0:
        return torch.roll(x, shifts=shift_size, dims=1)
    return x

def gilbert_shift_reverse(x, shift_size):
    """
    Reverse shift Gilbert sequence tensor
    Args:
        x: Shifted Gilbert sequence tensor, shape (B, N, C)
        shift_size: Shift size
    Returns:
        Reverse shifted Gilbert sequence tensor, shape (B, N, C)
    """
    if shift_size > 0:
        return torch.roll(x, shifts=-shift_size, dims=1)
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with global relative position bias.
    Use global 2D relative position bias, instead of window-based relative position bias.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        input_resolution (tuple[int]): Input resolution (H, W) for global RPB
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    _relative_position_bias_cache = {}

    def __init__(self, dim, window_size, num_heads, input_resolution, 
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., shift_size=0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.input_resolution = input_resolution  # H, W
        self.shift_size = shift_size
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # define global relative position bias table
        H, W = input_resolution
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * H - 1) * (2 * W - 1), num_heads))  # (2*H-1) * (2*W-1), nH
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def _calc_relative_position_index(self, positions_2d, H, W):
        """Calculate relative position index"""
        B_, N = positions_2d.shape[0], positions_2d.shape[1]
        
        # calculate relative coordinates
        relative_coords_h = positions_2d[:, :, 0].unsqueeze(2) - positions_2d[:, :, 0].unsqueeze(1)  # (B_, N, N)
        relative_coords_w = positions_2d[:, :, 1].unsqueeze(2) - positions_2d[:, :, 1].unsqueeze(1)  # (B_, N, N)

        relative_coords_h = relative_coords_h + H - 1
        relative_coords_w = relative_coords_w + W - 1
        
        relative_position_index = relative_coords_h * (2 * W - 1) + relative_coords_w  # (B_, N, N)
        
        return relative_position_index
    
    def _get_relative_position_bias(self, H, W, shift_size, device):
        """Get relative position bias - only cache index, recalculate bias each time"""
        cache_key = (H, W, self.window_size[0], self.window_size[1], shift_size)
        
        if cache_key not in self._relative_position_bias_cache:
            # calculate relative position index (this part does not depend on trainable parameters, can be cached)
            window_size_sq = self.window_size[0] * self.window_size[1]
            path_info = _global_gilbert_cache.get_or_create_path(H, W)
            reverse_map = path_info['reverse_map']
            
            num_windows = (H * W) // window_size_sq
            
            if shift_size > 0:
                positions_2d = reverse_map[:num_windows * window_size_sq]
                gilbert_shift = window_size_sq // 2
                positions_2d = torch.roll(positions_2d, shifts=gilbert_shift, dims=0)
                positions_2d = positions_2d.view(num_windows, window_size_sq, 2)
            else:
                positions_2d = reverse_map[:num_windows * window_size_sq].view(num_windows, window_size_sq, 2)
            
            relative_position_index = self._calc_relative_position_index(positions_2d, H, W)
            
            # only cache index, use detach() to break the computation graph
            self._relative_position_bias_cache[cache_key] = relative_position_index.detach()
        
        # get index from cache, ensure on correct device
        cached_index = self._relative_position_bias_cache[cache_key]
        if cached_index.device != device:
            cached_index = cached_index.to(device)
            # update device version in cache
            self._relative_position_bias_cache[cache_key] = cached_index
        
        # recalculate bias each time, ensure using latest trainable parameters
        bias_h = 2 * H - 1
        bias_w = 2 * W - 1
        relative_position_bias_table = self.relative_position_bias_table[:bias_h * bias_w]
        relative_position_bias = relative_position_bias_table[cached_index.long()]
        relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous()
        
        return relative_position_bias
    
    def forward(self, x, H, W, shift_size=0, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self._get_relative_position_bias(H, W, shift_size, x.device)
        
        num_windows = relative_position_bias.shape[0]
        
        if B_ % num_windows != 0:
            raise ValueError(f"Input batch dimension B_({B_}) must be divisible by window number ({num_windows})."
                           f"This usually indicates a problem with the batch_size or window splitting of the input data.")
        
        batch_size = B_ // num_windows
        if batch_size > 1:
            relative_position_bias = relative_position_bias.repeat(batch_size, 1, 1, 1)  # (B_, nH, N, N)
        attn = attn + relative_position_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
        attn_type (int): Type of attention to use. 0: Window attention, 1: Horizontal cross window attention, 2: Vertical cross window attention
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
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
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            input_resolution=self.input_resolution, shift_size=self.shift_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # create Gilbert shift attention mask
        if self.shift_size > 0:
            # calculate Gilbert shift size
            gilbert_shift = self.window_size * self.window_size // 2
            window_size_sq = self.window_size * self.window_size
            H, W = self.input_resolution
            total_positions = H * W
            num_windows = total_positions // window_size_sq
            
            # create shift mask (only one)
            attn_mask = self._create_gilbert_shift_mask(num_windows, window_size_sq, gilbert_shift)
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.register_buffer("attn_mask", None)
        self.fused_window_process = fused_window_process

    def _create_gilbert_shift_mask(self, num_windows, window_size_sq, gilbert_shift):
        """
        Create Gilbert shift attention mask
        Args:
            num_windows: Number of windows
            window_size_sq: Square of window size
            gilbert_shift: Gilbert shift size
        Returns:
            Attention mask, shape (num_windows, window_size_sq, window_size_sq)
        """
        H, W = self.input_resolution
        total_positions = H * W
        
        attn_mask = torch.zeros(num_windows, window_size_sq, window_size_sq, dtype=torch.float32)
        
        # use forward shift
        effective_shift = gilbert_shift
        
        # find the window that actually occurs wrap-around
        for w in range(num_windows):
            # calculate the range of the original sequence positions corresponding to this window after shift
            start_pos = (w * window_size_sq - effective_shift) % total_positions
            end_pos = ((w + 1) * window_size_sq - 1 - effective_shift) % total_positions
            
            # if start_pos > end_pos, it means wrap-around occurred
            if start_pos > end_pos:
                # this window contains the original sequence [start_pos, total_positions-1] and [0, end_pos]
                # need to calculate the positions of these two parts in the current window
                
                # front part: the position of the original sequence [start_pos, total_positions-1] in the window
                front_size = total_positions - start_pos
                
                # back part: the position of the original sequence [0, end_pos] in the window  
                back_size = end_pos + 1
                
                # these two parts are not continuous in 2D space, should be masked mutually
                if front_size > 0 and back_size > 0 and front_size + back_size == window_size_sq:
                    attn_mask[w, :front_size, front_size:] = float('-inf')
                    attn_mask[w, front_size:, :front_size] = float('-inf')
        
        return attn_mask

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        
        gilbert_shift = self.window_size * self.window_size // 2 if self.shift_size > 0 else 0
        path_info = _global_gilbert_cache.get_or_create_path(H, W)
        reverse_map = path_info['reverse_map'].to(x.device)  # (H*W, 2)
        if self.shift_size > 0:
            x_shifted = gilbert_shift_forward(x, gilbert_shift)
            x_windows = gilbert_tensor_to_windows(x_shifted, self.window_size, reverse_map, self.shift_size)
            attn_windows = self.attn(x_windows, H, W, self.shift_size, mask=self.attn_mask)
            attn_gilbert = gilbert_windows_to_tensor(attn_windows, B, H, W)
            attn_gilbert = gilbert_shift_reverse(attn_gilbert, gilbert_shift)
        else:
            x_windows = gilbert_tensor_to_windows(x, self.window_size, reverse_map, self.shift_size)
            attn_windows = self.attn(x_windows, H, W, self.shift_size, mask=None)
        
        x = shortcut + self.drop_path(attn_gilbert)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

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

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, 
                input_resolution=input_resolution,
                num_heads=num_heads, 
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,  # 奇数层使用Gilbert移位
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop, 
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                fused_window_process=fused_window_process
            )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        if len(x.shape) == 4:
            B, H, W, C = x.shape
        elif len(x.shape) == 3:
            B, L, C = x.shape
            H, W = self.input_resolution
            assert L == H * W, f"Input sequence length {L} does not match resolution {H}x{W}"
            x = x.view(B, H, W, C)
        else:
            raise ValueError(f"Unsupported input tensor dimension: {x.shape}")
        # convert to Gilbert sequence format for processing
        x_gilbert = tensor_to_gilbert_path(x, cache=_global_gilbert_cache)  # (B, H*W, C)

        for blk in self.blocks:
            if self.use_checkpoint:
                x_gilbert = checkpoint.checkpoint(blk, x_gilbert)
            else:
                x_gilbert = blk(x_gilbert)
        x_2d = gilbert_tensor_to_2d(x_gilbert, H, W, cache=_global_gilbert_cache)  # (B, H, W, C)
 
        if self.downsample is not None:
            x_flat = x_2d.view(B, H * W, C)
            x_2d = self.downsample(x_flat)   # downsample returns (B, H/2*W/2, 2*C)
            return x_2d
        else:
            return x_2d.view(B, H * W, C)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[3, 3, 6, 3], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
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

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

