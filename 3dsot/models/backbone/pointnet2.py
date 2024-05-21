import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2.utils import pointnet2_utils
from pointnet2.utils import pytorch_utils as pt_utils
        

class PointnetSetAbstraction(nn.Module):
    def __init__(self, radius, num_samples, mlps, use_fps):
        super(PointnetSetAbstraction, self).__init__()
        self.use_fps = use_fps

        self.grouper = pointnet2_utils.QueryAndGroup(radius, num_samples)
        self.conv_layers = []
        self.batch_norm_layers = []
        self.in_channel = mlps[0]+3

        for out_channel in mlps[1:]:
            self.conv_layers.append(
                nn.Conv2d(self.in_channel, out_channel, kernel_size=1).cuda()
            )
            self.batch_norm_layers.append(nn.BatchNorm2d(num_features=out_channel).cuda())
            self.in_channel = out_channel
        

    def forward(self, xyz, features, num_points):
        batch_size = xyz.shape[0]
        self.num_points = num_points

        if self.use_fps:
            sample_idxs = pointnet2_utils.furthest_point_sample(xyz, self.num_points)
        else:
            sample_idxs = torch.arange(self.num_points).repeat(batch_size, 1).int().cuda()

        xyz_flipped = xyz.permute(0, 2, 1).contiguous()
        
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, sample_idxs).permute(0, 2, 1).contiguous()
        new_features = self.grouper(xyz, new_xyz, features)

        for idx, conv_layer in enumerate(self.conv_layers):
            batch_norm = self.batch_norm_layers[idx]
            new_features = nn.Sequential(conv_layer, batch_norm, nn.ReLU(inplace=True))(new_features)

        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.shape[-1]]).squeeze(-1)

        torch.cuda.empty_cache()
        return new_xyz, new_features, sample_idxs


class Pointnet_Plus_Plus(nn.Module):
    def __init__(self, use_fps, normalize_xyz, return_intermediate=False):
        super(Pointnet_Plus_Plus, self).__init__()
        self.use_fps = use_fps
        self.normalize_xyz = normalize_xyz
        self.return_intermediate = return_intermediate

        self.num_set_abstraction_modules = 3
        self.mlps = [
            [0, 64, 64, 128],
            [128, 128, 128, 256],
            [256, 256, 256, 256]
        ]
        self.radii = [0.3, 0.5, 0.7]
        self.num_samples = [32, 32, 32]

        self.modules = []
        for idx in range(self.num_set_abstraction_modules):
            module = PointnetSetAbstraction(
                radius=self.radii[idx],
                num_samples=self.num_samples[idx],
                mlps=self.mlps[idx],
                use_fps=use_fps
            )
            self.modules.append(module)
        
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features
        
    def forward(self, pointcloud, npoints):
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features, l_idx = [xyz], [features], []
        for module_idx, module in enumerate(self.modules):
            xyz, features, idx = module(xyz, features, npoints[module_idx])
            l_xyz.append(xyz)
            l_features.append(features)
            l_idx.append(idx)
        if self.return_intermediate:
            return l_xyz[1:], l_features[1:], l_idx[0]
        return xyz, features, l_idx[0]
