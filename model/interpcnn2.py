from .interpcnn2_utils import square_distance, farthest_point_sample, index_points, uniform_point_sample
from .interpcnn2_utils import BatchMatMul
from qqquantize.qmodules import InputStub

import torch
import torch.nn as nn
import torch.nn.functional as F

NORM_TYPE = 'bn'

class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    def forward(self, inputs):
        if isinstance(inputs, torch.Tensor):
            return inputs.permute(self.dims)
        else:
            return [x.permute(d) for x,d in zip(inputs, self.dims)]

class ConcatXYZ(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs):
        xyz, data = inputs # [BS, N, C]
        data = torch.cat([data, xyz], dim=-1)
        return xyz, data

class PointDownsample(nn.Module):
    def __init__(self, n=None, ratio=None, interpolation='fps'):
        super().__init__()
        assert bool(n) ^ bool(ratio), f"only one of n and ratio can be True, but get n={n}, ratio={ratio}"
        self.n = n
        self.ratio = ratio
        self.interpolation = interpolation

        self._local_index = None

    def forward(self, args):
        xyz, data = args
        BS, N, C = xyz.shape
        if self.ratio:
            n = int(N * self.ratio)
        else:
            n = self.n
        """if self.interpolation == 'uniform':
            idx = uniform_point_sample(xyz, n)
        elif self.interpolation == 'fps':
            idx = farthest_point_sample(xyz, n)
        self._local_index = idx
        new_xyz = index_points(xyz, idx)
        new_data = index_points(data, idx)
        """
        new_xyz=torch.narrow(xyz,1,0,xyz.shape[1]//4)
        new_data=torch.narrow(data,1,0,data.shape[1]//4)
        
        return new_xyz, new_data

class CheckNpoints(nn.Module):
    def __init__(self, npoints):
        super().__init__()
        self.npoints = npoints
    def forward(self, args):
        xyz, data = args
        BS, N, C = xyz.shape
        assert N == self.npoints, f"CheckNpoints, expect {self.npoints} points, but {N}"
        return xyz, data

class PointLBR(nn.Module):
    def __init__(self, in_channels, out_channels, norm=NORM_TYPE, relu=True, cat_xyz=False):
        super().__init__()
        self.cat_xyz = cat_xyz
        if cat_xyz:
            in_channels += 3
        self.linear = nn.Linear(in_channels, out_channels, bias=not norm)
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(8, out_channels)
        else:
            self.norm = None
        if relu:
            self.relu = nn.ReLU()
        
    def forward(self, args):
        """
        x: [BS, npoint, dim]
        """
        xyz, data = args
        if self.cat_xyz:
            data = torch.cat([data, xyz], -1)
        data = self.linear(data)
        if self.norm:
            data = data.permute([0, 2, 1])
            data = self.norm(data)
            data = data.permute([0, 2, 1])
        if hasattr(self, 'relu'):
            data = self.relu(data)
        return xyz, data

class CheckNpoints(nn.Module):
    def __init__(self, npoints, name=None):
        super().__init__()
        self.npoints = npoints
        self.name = name
    def forward(self, args):
        xyz, data = args
        BS, N, C = xyz.shape
        assert N == self.npoints, f"CheckNpoints {self.name}, expect {self.npoints} points, but {N}"
        return xyz, data

class PointMaxPooling(nn.Module):
    def __init__(self, local_size, is_global=False):
        super().__init__()
        assert not (local_size and is_global), "local_size and is_global cannot all be True"
        self.local_size = local_size
        self.is_global = is_global

        self._local_index = None

    def forward(self, args):
        xyz, data = args
        device = xyz.device
        BS, N, C = xyz.shape
        if self.is_global:
            data = torch.max(data, 1)[0].unsqueeze(1)
            xyz = torch.zeros([BS, 1, C], device=xyz.device)
            return xyz, data
        else:
            dist = square_distance(xyz, xyz) # [B, N, N]
            dist=dist.to(xyz.device)
            dist[dist<1e-8] = float("Inf")
            _, local_index = torch.topk(dist, self.local_size, dim=-1, largest=False, sorted=False) # [B, npoint, self.local_size]
            self_index = torch.arange(N, device=device, dtype=local_index.dtype).reshape([1, N, 1]).repeat(BS, 1, 1)
            local_index = torch.cat([self_index, local_index], dim=-1)
            self._local_index = local_index

            local_data = index_points(data, local_index) # [BS, N, self.local_size, C]
            local_data, _ = torch.max(local_data, dim=2)
            return xyz, local_data

class InterPConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, local_size, depthwise=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.local_size = local_size
        self.depthwise = depthwise
        self.interp = nn.Sequential(
            nn.Linear(3, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, kernel_size),
            nn.Softmax(-1)
        )
        self.bmm = BatchMatMul()

        if self.depthwise:
            assert in_channels == out_channels, 'depthwise conv require same channels for input and output'
            self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, groups=1)
        
        # for export
        self._local_index = None

    def forward(self, args):
        """
        x: [BS, npoint, dim]
        """
        xyz, data = args
        BS, N, dim = data.shape
        assert dim == self.in_channels, f"expect dim={self.in_channels}, but {dim}"
        assert xyz.shape[0] == BS and xyz.shape[1] == N
        ks = self.kernel_size
        local_size = self.local_size

        # find local points
        dist = square_distance(xyz, xyz) # [B, N, N]
        dist=dist.to(xyz.device)
        dist[dist<1e-8] = float("Inf")
        _, local_index = torch.topk(dist, local_size-1, dim=-1, largest=False, sorted=False) # [B, N, local_size - 1]
        self_index = torch.arange(N, dtype=local_index.dtype, device=local_index.device)
        self_index = self_index.reshape([1, N, 1]).repeat(BS, 1, 1)
        local_index = torch.cat([self_index, local_index], dim=-1)
        self._local_index = local_index
        
        local_data = index_points(data, local_index) # [B, N, local_size, C]
        local_xyz = index_points(xyz, local_index)
        
        # calc alpha
        relative_xyz = local_xyz - xyz.unsqueeze(2)
        relative_xyz = relative_xyz.reshape(BS*N*local_size, 3)
        alpha = self.interp(relative_xyz).reshape([BS*N, local_size, ks]) # [B*N, local_size, kernel_size]
        
        # convolution
        local_data = local_data.reshape(BS*N, local_size, dim).permute([0, 2, 1]) # [B*N, dim. local_size]
        local_data = self.bmm(local_data, alpha) # [BS*N, dim, ks]
        out = self.conv(local_data)
        assert out.shape[-1] == 1
        out = out.reshape([BS, N, -1])
        return xyz, out

class InterpCNN2(nn.Module):
    def __init__(self, args):
        super().__init__()
        CAT_XYZ = True


        self.net = nn.Sequential(
            Permute([(0, 2, 1), (0, 2, 1)]),
            
            CheckNpoints(1024, 1),
            InterPConv(1, 8, 32, 32, depthwise=False),
            PointLBR(8, 32, cat_xyz=CAT_XYZ),
            PointMaxPooling(32),
            PointDownsample(ratio=0.25),
            CheckNpoints(256, 2),

            PointLBR(32, 64, cat_xyz=CAT_XYZ),
            InterPConv(64, 64, 16, 16, depthwise=True),
            PointLBR(64, 64, cat_xyz=CAT_XYZ),
            PointMaxPooling(16),
            PointDownsample(ratio=0.25),
            CheckNpoints(64, 4),

            PointLBR(64, 128, cat_xyz=CAT_XYZ),
            InterPConv(128, 128, 16, 16, depthwise=True),
            PointLBR(128, 128, cat_xyz=CAT_XYZ),
            CheckNpoints(64, 5),
            
            Permute([(0, 2, 1), (0, 2, 1)]),
        )
        self.fc = nn.Linear(128, 40)

        self.input_stub = InputStub()
    
    def forward(self, points):
        points = self.input_stub(points)
        B, dim, N = points.shape
        init_data = torch.ones([B, 1, N], device=points.device, dtype=points.dtype)
        _, x = self.net((points, init_data))
        x = torch.mean(x, -1)
        logits = self.fc(x)
        return logits
