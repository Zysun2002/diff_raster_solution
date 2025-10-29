import torch
import math
import numpy as np
import torch
import math
import ipdb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from .share import sh
import torch.nn.functional as F


class SmoothnessLoss:
    def __init__(self, setting):
        #0
        self.setting = setting
        self.mode = setting['mode']
        self.w = setting["w"]
 
        # self.w_0 = 1
        # self.w_1 = 0.5
        # self.w_2 = 0.5
        # self.w_3 = 0.5
        
        # self.w_4 = 0.5
        # self.reg_4 = 0.5

        # self.w_5 = 0.5
        # self.w_6 = 0.5

    def __call__(self, points, points_init=None, is_close=None, ext_w=None):
        if self.mode is None:return None

        
        if self.mode == "poisson_and_angle":
            return self.forward_poisson_and_angle(points, points_init, is_close)
        elif self.mode == "angle_weighted":
            return self.forward_angle_weighted(points, points_init, is_close)
        elif self.mode == "external_weight":
            return self.forward_external_w(points, ext_w, points_init, is_close)
        else:
            raise ValueError(f"Unknown forward mode: {self.mode}")
    
    def latex(self):
        if self.mode == "poisson_and_angle":
            return self.latex_poisson_and_angle()
        elif self.mode == "angle_weighted":
            return self.latex_angle_weighted()
        # Add more elif branches for other modes
        else:
            raise ValueError(f"Unknown latex mode: {self.mode}")

    def forward_external_w(self, points, ext_w, points_init,is_close):

        if is_close:
            points = torch.cat([points[-1:], points, points[:1]], dim=0)
            points_init = torch.cat([points_init[-1:], points_init, points_init[:1]], dim=0)

        e1 = points[:-2] - points[1:-1]
        e2 = points[1:-1] - points[2:]

        e1_init = points_init[:-2] - points_init[1:-1]
        e2_init = points_init[1:-1] - points_init[2:]

        e1_init_norm = e1 / (e1_init.norm(dim=1, keepdim=True) + 1e-6)
        e2_init_norm = e2 / (e2_init.norm(dim=1, keepdim=True) + 1e-6)

        diff = e1_init_norm - e2_init_norm
        
        return self.w * (ext_w * (diff ** 2).mean(dim=1)).mean()


    def forward_0(self, points, is_close):
        """direction and distance"""      

        if is_close:
            points = torch.cat([points, points[:2]], dim=0)

        diff = points[:-2] - 2 * points[1:-1] + points[2:]
        return self.w_0 * (diff ** 2).mean()
    
    def forward_edge_norm(self, points, is_close):
        """normalize by edge distance"""
        
        if is_close:
            points = torch.cat([points, points[:2]], dim=0)
        # points: tensor of shape (N, D)
        # diff = points[:-2] - 2 * points[1:-1] + points[2:]  # second-order diff
        
        e1 = points[:-2] - points[1:-1]
        e2 = points[1:-1] - points[2:]

        e1_norm = e1 / (e1.norm(dim=1, keepdim=True) + 1e-6)
        e2_norm = e2 / (e2.norm(dim=1, keepdim=True) + 1e-6)

        diff = e1_norm - e2_norm
        return self.w_1 * (diff ** 2).mean()
    
    def latex_edge_norm(self):
        return fr"L_{{smooth}} = {self.w_1} \cdot  \frac{{1}}{{N}} \sum_i \left\| \frac{{p_{{i-1}} - p_i}}{{\|p_{{i-1}} - p_i\|}} - \frac{{p_i - p_{{i+1}}}}{{\|p_i - p_{{i+1}}\|}} \right\|^2"

    def forward_L1(self, points, points_init, is_close):
        """L1 version of edge-normalized smoothness loss"""
        if is_close:
            points = torch.cat([points, points[:2]], dim=0)
            points_init = torch.cat([points_init, points_init[:2]], dim=0)

        e1 = points[:-2] - points[1:-1]
        e2 = points[1:-1] - points[2:]

        e1_init = points_init[:-2] - points_init[1:-1]
        e2_init = points_init[1:-1] - points_init[2:]

        e1_init_norm = e1 / (e1_init.norm(dim=1, keepdim=True) + 1e-6)
        e2_init_norm = e2 / (e2_init.norm(dim=1, keepdim=True) + 1e-6)

        diff = e1_init_norm - e2_init_norm
        return self.w_1 * (diff.abs()).mean()
    
    def latex_L1(self):
        return fr"L_{{smooth}} = {self.w_2} \cdot  \frac{{1}}{{N}} \sum_i \left\| \frac{{p_{{i-1}} - p_i}}{{\|p_{{i-1}}^0 - p_i^0\|}} - \frac{{p_i - p_{{i+1}}}}{{\|p_i^0 - p_{{i+1}}^0\|}} \right\|_1"

    def forward_init_edge_norm(self, points, points_init, is_close):
        """normalize by initial edge distance"""
        if is_close:
            points = torch.cat([points, points[:2]], dim=0)
            points_init = torch.cat([points_init, points_init[:2]], dim=0)

        e1 = points[:-2] - points[1:-1]
        e2 = points[1:-1] - points[2:]

        e1_init = points_init[:-2] - points_init[1:-1]
        e2_init = points_init[1:-1] - points_init[2:]

        e1_init_norm = e1 / (e1_init.norm(dim=1, keepdim=True) + 1e-6)
        e2_init_norm = e2 / (e2_init.norm(dim=1, keepdim=True) + 1e-6)

        diff = e1_init_norm - e2_init_norm
        return self.w_2 * (diff ** 2).mean()
    
    def latex_init_edge_norm(self):
        return fr"L_{{smooth}} = {self.w_3} \cdot  \frac{{1}}{{N}} \sum_i \left\| \frac{{p_{{i-1}} - p_i}}{{\|p_{{i-1}}^0 - p_i^0\|}} - \frac{{p_i - p_{{i+1}}}}{{\|p_i^0 - p_{{i+1}}^0\|}} \right\|^2"

    def forward_init_edge_regularization(self, points, points_init, is_close):
        """normalize by initial edge distance + regularization to initial points"""

        if is_close:
            points = torch.cat([points, points[:2]], dim=0)
            points_init = torch.cat([points_init, points_init[:2]], dim=0)

        e1 = points[:-2] - points[1:-1]
        e2 = points[1:-1] - points[2:]

        e1_init = points_init[:-2] - points_init[1:-1]
        e2_init = points_init[1:-1] - points_init[2:]

        e1_init_norm = e1 / (e1_init.norm(dim=1, keepdim=True) + 1e-6)
        e2_init_norm = e2 / (e2_init.norm(dim=1, keepdim=True) + 1e-6)

        diff = e1_init_norm - e2_init_norm

        regu = (e1 - e1_init) ** 2


        return self.w_4 * (diff ** 2).mean() + self.reg_4 * regu.mean()
    
    def latex_init_edge_regularization(self):
        return fr"L_{{smooth}} = {self.w_4} \cdot  \frac{{1}}{{N}} \sum_i \left\| \frac{{p_{{i-1}} - p_i}}{{\|p_{{i-1}}^0 - p_i^0\|}} - \frac{{p_i - p_{{i+1}}}}{{\|p_i^0 - p_{{i+1}}^0\|}} \right\|^2 + {self.reg_4} \cdot \frac{{1}}{{N}} \sum_i \| (p_{{i+1}} - p_{{i}}) - (p_{{i+1}}^0 - p_{{i}}^0) \|^2"

    def forward_edge_weighted(self, points, points_init, is_close):
        """weighted version of edge-normalized smoothness loss"""
        
        N = points.shape[0]

        if is_close:
            points = torch.cat([points[-1:], points, points[:1]], dim=0)
            points_init = torch.cat([points_init[-1:], points_init, points_init[:1]], dim=0)

        e1 = points[:-2] - points[1:-1]
        e2 = points[1:-1] - points[2:]

        e1_init = points_init[:-2] - points_init[1:-1]
        e2_init = points_init[1:-1] - points_init[2:]

        e1_init_norm = e1 / (e1_init.norm(dim=1, keepdim=True) + 1e-6)
        e2_init_norm = e2 / (e2_init.norm(dim=1, keepdim=True) + 1e-6)

        diff = e1_init_norm - e2_init_norm
        weights_tensor = N * (e1.norm(dim=1) + e2.norm(dim=1)) / (2 * e1.norm(dim=1).sum())
        # ipdb.set_trace()
        return self.w_5 * (weights_tensor * (diff ** 2).mean(dim=1)).mean()
    
    def latex_edge_weighted(self):
        return fr"L_{{smooth}} = {self.w_5} \cdot  \frac{{1}}{{N}} \sum_i w_i \left\| \frac{{p_{{i-1}} - p_i}}{{\|p_{{i-1}}^0 - p_i^0\|}} - \frac{{p_i - p_{{i+1}}}}{{\|p_i^0 - p_{{i+1}}^0\|}} \right\|^2, \quad w_i = N \cdot \frac{{\|p_{{i-1}} - p_i\| + \|p_i - p_{{i+1}}\|}}{{2 \sum_j \|p_j - p_{{j+1}}\|}}"

    def forward_angle_weighted(self, points, points_init, is_close):
        """weighted version of angle-based smoothness loss"""
        
        eps = 1e-8
        N = points.shape[0]

        if is_close:
            points = torch.cat([points[-1:], points, points[:1]], dim=0)
            points_init = torch.cat([points_init[-1:], points_init, points_init[:1]], dim=0)

        e1 = points[:-2] - points[1:-1]
        e2 = points[1:-1] - points[2:]

        e1_init = points_init[:-2] - points_init[1:-1]
        e2_init = points_init[1:-1] - points_init[2:]

        e1_init_norm = e1 / (e1_init.norm(dim=1, keepdim=True) + 1e-6)
        e2_init_norm = e2 / (e2_init.norm(dim=1, keepdim=True) + 1e-6)

        e1_unit = e1 / (e1.norm(dim=1, keepdim=True) + eps)
        e2_unit = e2 / (e2.norm(dim=1, keepdim=True) + eps)
        cos_theta = (e1_unit * e2_unit).sum(dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

        cos_135 = -0.7071067
        diff = e1_init_norm - e2_init_norm
        weights_tensor = 0.55 + 0.45 * (-torch.tanh(100*(cos_theta-cos_135))+1)
        

        return self.w * (weights_tensor * (diff ** 2).mean(dim=1)).mean()

    def latex_angle_weighted(self):
        return fr"L_{{smooth}} = {self.w} \cdot  \frac{{1}}{{N}} \sum_i w_i \left\| \frac{{p_{{i-1}} - p_i}}{{\|p_{{i-1}}^0 - p_i^0\|}} - \frac{{p_i - p_{{i+1}}}}{{\|p_i^0 - p_{{i+1}}^0\|}} \right\|^2, \quad w_i = 0.55 + 0.45 \cdot (-tanh(100 \cdot (\cos{{x}}-\cos{{135}}))+1)"

    def forward_poisson_and_angle(self, points, points_init, is_close):
        """weighted version of angle-based smoothness loss"""
        
        eps = 1e-8
        N = points.shape[0]

        if is_close:
            points = torch.cat([points[-1:], points, points[:1]], dim=0)
            points_init = torch.cat([points_init[-1:], points_init, points_init[:1]], dim=0)

        e1 = points[:-2] - points[1:-1]
        e2 = points[1:-1] - points[2:]

        e1_init = points_init[:-2] - points_init[1:-1]
        e2_init = points_init[1:-1] - points_init[2:]

        e1_init_norm = e1 / (e1_init.norm(dim=1, keepdim=True) + 1e-6)
        e2_init_norm = e2 / (e2_init.norm(dim=1, keepdim=True) + 1e-6)

        e1_unit = e1 / (e1.norm(dim=1, keepdim=True) + eps)
        e2_unit = e2 / (e2.norm(dim=1, keepdim=True) + eps)
        cos_theta = (e1_unit * e2_unit).sum(dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

        cos_135 = -0.7071067
        diff = e1_init_norm - e2_init_norm
        weights_tensor = 0.55 +0.45 * (-torch.tanh(100*(cos_theta-cos_135))+1)

        regu = (e1 - e1_init) ** 2
        loss_poison = self.setting["reg_w"] * regu.mean()

        loss_angle = self.w * (weights_tensor * (diff ** 2).mean(dim=1)).mean()


        return loss_angle + loss_poison
    
    def latex_poisson_and_angle(self):
        return fr"L_{{smooth}} = {self.w} \cdot  \frac{{1}}{{N}} \sum_i w_i \left\| \frac{{p_{{i-1}} - p_i}}{{\|p_{{i-1}}^0 - p_i^0\|}} - \frac{{p_i - p_{{i+1}}}}{{\|p_i^0 - p_{{i+1}}^0\|}} \right\|^2 + {self.setting['reg_w']} \cdot \frac{{1}}{{N}} \sum_i \| (p_{{i+1}} - p_{{i}}) - (p_{{i+1}}^0 - p_{{i}}^0) \|^2 , \quad w_i = 0.55 + 0.45 \cdot (-tanh(100 \cdot (\cos{{x}}-\cos{{135}}))+1)"



class BandLoss:
    def __init__(self, setting):
        
        self.mode = setting['mode']
        self.w = setting["w"]
        # self.w_0 = 100

    def __call__(self, points, udf):
        if self.mode is None:
            print("No band loss applied.")
            return torch.tensor(0.0, dtype=torch.float32, device=points.device)

        if self.mode == "midpoint":
            return self.forward_1(points, udf)
        else:
            raise ValueError(f"Unknown forward mode: {self.mode}")
        # return self.forward_1(points, udf)

    def forward_0(self, points, udf):

        # points = points.clone()

        udf = udf.unsqueeze(0).unsqueeze(0)

        grid = points * 2 - 1   # (N, 2)
        grid = grid.unsqueeze(0).unsqueeze(2)  # reshape for grid_sample

        # F.grid_sample considers the offset of pixel center
        sampled = F.grid_sample(udf, grid, align_corners=False)
        sampled = sampled.view(-1)  # (N,)

        # Loss = squared distance to 0 band
        loss = self.w * (sampled ** 2).mean()
        return loss

    def forward_1(self, points, udf, is_close=True):
        """enforce midpoints in the band"""
        if is_close:
            points_closed = torch.cat([points, points[:1]], dim=0)

        midpoints = (points_closed[:-1] + points_closed[1:]) / 2

        points_loss = self.forward_0(points, udf)
        midpoints_loss = self.forward_0(midpoints, udf)

        return points_loss + midpoints_loss
    
    def latex_1(self):
        return fr"L_{{band}} = {self.w} \cdot \frac{{1}}{{N}} \sum_i  (\text{{UDF}}(p_i) + \text{{UDF}}(mid point_i))"


class ImageLoss:
    def __init__(self):
        self.w = 100.

    def __call__(self, img, target):
        return self.forward(img, target)

    def forward(self, img, target):
        return self.w * (img - target).pow(2).mean()
    
    def latex(self):
        return fr"L_{{img}} = {self.w} \cdot \frac{1}{{N}} \sum_i (I_i - T_i)^2"


class StraightnessLoss:
    def __init__(self, setting):

        self.mode = setting['mode']
        self.w = setting["w"]


        # self.w = 0.1
        self.alpha = 1000
        self.beta = 0.95
        self.idx = 0.5

    def __call__(self, points, is_close):

        if self.mode is None:return torch.tensor(0.0, dtype=torch.float32, device=points.device, requires_grad=True)

        if self.mode == "tanh":
            return self.forward(points, is_close)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def forward(self, points, is_close, eps=1e-8):
        if is_close:
            points = torch.cat([points, points[:2]], dim=0)

        n = points.shape[0]
        if n < 3:
            return torch.tensor(0.0, dtype=torch.float32, device=points.device)

        # vectors: (N-2, 2)
        ba = points[:-2] - points[1:-1]
        bc = points[2:]  - points[1:-1]

        # normalize
        ba = ba / (ba.norm(dim=1, keepdim=True) + eps)
        bc = bc / (bc.norm(dim=1, keepdim=True) + eps)

        # cosine similarity
        cos_theta = (ba * bc).sum(dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

        # peak in radians

        # scale factor to control how fast it drops after peak


        penalties = (-torch.tanh(self.alpha * (cos_theta + self.beta)) + 1) * ((cos_theta+1) ** self.idx)
        # penalties =  ((cos_theta + 1) ** 2) * 0.1

        return self.w * penalties.mean()
    
    def latex(self):
        return fr"L_{{straight}} = {self.w} \cdot \frac{{1}}{{N}} \sum_i \left( -\tanh({self.alpha} (\cos \theta_i + {self.beta})) + 1 \right) \cdot (\cos \theta_i + 1)^{{{self.idx}}}"
        

def cal_band_indicator_loss(points, udf, eps=1e-6):
    """
    Points: (N, 2), normalized to [0,1].
    UDF: (H, W) torch tensor.
    Returns: scalar sum of 0/1 loss.
    """
    # Prepare UDF for grid_sample
    udf = udf.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # Convert [0,1] → [-1,1] for grid_sample
    grid = points * 2 - 1   # (N, 2)
    grid = grid.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)

    # Bilinear sample UDF at point locations
    sampled = F.grid_sample(udf, grid, align_corners=True)
    sampled = sampled.view(-1)  # (N,)

    # Indicator loss: 0 if ~0, else 1
    loss = (sampled.abs() > eps).float().sum()
    return loss


def cal_straightness_loss(points, is_close, peak_angle=178, eps=1e-8):
    """
    Straightness loss using tanh: nearly constant before peak_angle,
    drops smoothly toward 180 degrees.
    """
    if is_close:
        points = torch.cat([points, points[:2]], dim=0)


    n = points.shape[0]
    if n < 3:
        return torch.tensor(0.0, dtype=torch.float32, device=points.device)

    # vectors: (N-2, 2)
    ba = points[:-2] - points[1:-1]
    bc = points[2:]  - points[1:-1]

    # normalize
    ba = ba / (ba.norm(dim=1, keepdim=True) + eps)
    bc = bc / (bc.norm(dim=1, keepdim=True) + eps)

    # cosine similarity
    cos_theta = (ba * bc).sum(dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

    # peak in radians

    # scale factor to control how fast it drops after peak
    alpha = 100.0

    penalties = (-torch.tanh(alpha * (cos_theta + 0.99999)) + 1) * ((cos_theta+1) ** 0.3)
    # penalties =  ((cos_theta + 1) ** 2) * 0.1

    return penalties.mean()


def cal_straightness_loss_v1(points, is_close, peak_angle=178, eps=1e-8):
    """
    Straightness loss using tanh: nearly constant before peak_angle,
    drops smoothly toward 180 degrees.
    """

    if is_close:
        points = torch.cat([points, points[:2]], dim=0)

    n = points.shape[0]
    if n < 3:
        return torch.tensor(0.0, dtype=torch.float32, device=points.device)

    # vectors: (N-2, 2)
    ba = points[:-2] - points[1:-1]
    bc = points[2:]  - points[1:-1]

    # normalize
    ba = ba / (ba.norm(dim=1, keepdim=True) + eps)
    bc = bc / (bc.norm(dim=1, keepdim=True) + eps)

    # cosine similarity
    cos_theta = (ba * bc).sum(dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

    # peak in radians
    peak_rad = torch.deg2rad(torch.tensor(peak_angle, dtype=torch.float32, device=points.device))
    cos_peak = torch.cos(peak_rad)  # corresponds to peak_angle

    # scale factor to control how fast it drops after peak
    alpha = 100.0

    # tanh loss: stays near 0.1 before peak, drops after
    penalties = 1 *  torch.tanh(alpha * (cos_theta - cos_peak))

    return penalties.mean() * 10000000


def cal_axis_align_loss(points, eps=1e-8):

    points = torch.cat([points, points[:1]], dim=0)

    # Compute edges
    edges = points[1:] - points[:-1]
    edges_norm = edges / (edges.norm(dim=1, keepdim=True) + eps)

    # assuming edges_norm is [N, 2]
    x_dir = torch.tensor([1.0, 0.0], device=edges_norm.device)
    y_dir = torch.tensor([0.0, 1.0], device=edges_norm.device)

    # inner products
    dot_x = (edges_norm * x_dir).sum(dim=1)  # [N]
    dot_y = (edges_norm * y_dir).sum(dim=1)  # [N]

    # absolute values
    abs_dot_x = dot_x.abs()
    abs_dot_y = dot_y.abs()

    # take elementwise minimum
    min_val = torch.max(abs_dot_x, abs_dot_y)  # [N]
    
    alpha = 50.0

    # tanh loss: stays near 0.1 before peak, drops after
    penalties = -0.01 * torch.tanh(alpha * (min_val - 0.9998)) + 0.01


    return penalties.mean()



def cal_curvature_loss(points, return_w=False):
    """
    Smooth differentiable curvature loss over a sequential list of 2D points.

    Args:
        points: (N,2) tensor of 2D points (in order)
        return_w: if True, also return the per-quadruple weights w
    """

    points = torch.cat([points, points[:3]], dim=0)

    # Hyperparameters (fixed inside function)
    l_cont   = 60   # continuity scaling factor
    b_infl   = 0.1  # inflection bias
    l_infl   = 90   # inflection offset
    sharpness = 20  # controls softness of transition

    n = points.shape[0]
    if n < 4:
        return (torch.tensor(0.0, dtype=torch.float32, device=points.device),
                torch.tensor([]) if return_w else None)

    # Consecutive quadruples
    pi, pj, pk, pl = points[:-3], points[1:-2], points[2:-1], points[3:]

    # Vectors
    vij = pj - pi
    vjk = pk - pj
    vkl = pl - pk

    # Normalize helper
    def normalize(v):
        return v / (v.norm(dim=1, keepdim=True) + 1e-6)

    vij_n = normalize(vij)
    vjk_n = normalize(vjk)
    vkl_n = normalize(vkl)

    # Angles (in degrees)
    cos1 = (vij_n * vjk_n).sum(dim=1).clamp(-0.9999, 0.9999)
    cos2 = (vjk_n * vkl_n).sum(dim=1).clamp(-0.9999, 0.9999)

    # 2D cross products 
    cross1 = vij[:, 0] * vjk[:, 1] - vij[:, 1] * vjk[:, 0]
    cross2 = vjk[:, 0] * vkl[:, 1] - vjk[:, 1] * vkl[:, 0]

    # Smooth weight in [0,1]
    sign_val = cross1 * cross2 + 1e-3
    w = torch.sigmoid(sharpness * sign_val)

    # Two candidate values
    same_side_val = torch.abs(cos1 - cos2) * 1
    inflection_val = - torch.min(cos1, cos2) * 1 + 5

    # Smooth interpolation
    val = w * same_side_val + (1 - w) * inflection_val

    if return_w:
        return 0.05 * val.mean(), w.detach().cpu().numpy()
    return 0.05 * val.mean()



def test_curvature_loss():
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import math, torch
    # Fixed first two edges to form ~135° at p1
    p1 = torch.tensor([0.0, 0.0])
    p2 = torch.tensor([1.0, 0.0])
    angle1 = math.radians(45.0)
    p0 = p1 + torch.tensor([math.cos(angle1), math.sin(angle1)])  # fixed first edge

    angles = list(range(0, 360, 1))
    losses, weights = [], []

    for ang in angles:
        ang_rad = math.radians(ang)
        p3 = p2 + torch.tensor([math.cos(ang_rad), math.sin(ang_rad)])
        pts = torch.stack([p0, p1, p2, p3], dim=0)
        loss, w = cal_curvature_loss(pts, return_w=True)
        losses.append(loss.item())
        weights.append(w[0])  # only one quadruple

    # Show only 8 polylines
    angles_show = range(0, 360, 45)

    # --- One figure with GridSpec ---
    fig = plt.figure(figsize=(14, 8))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 4, figure=fig)  # 3 rows × 4 cols

    # Big loss plot across top row
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_xlabel("Second angle (degrees)")
    ax1.set_ylabel("Curvature loss", color="tab:blue")
    ax1.plot(angles, losses, color="tab:blue", marker="o", markersize=2, label="Loss")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(45))

    ax2 = ax1.twinx()
    ax2.set_ylabel("Weight w", color="tab:red")
    ax2.plot(angles, weights, color="tab:red", linestyle="--", label="w")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_title("Curvature loss & weight vs. second angle")

    # Grid of 8 polylines in rows 2–3
    for i, ang in enumerate(angles_show):
        row = 1 + i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])

        ang_rad = math.radians(ang)
        p3 = p2 + torch.tensor([math.cos(ang_rad), math.sin(ang_rad)])
        pts = torch.stack([p0, p1, p2, p3], dim=0)

        x, y = pts[:, 0], pts[:, 1]
        ax.plot(x, y, "o-", linewidth=2)
        ax.set_title(f"{ang}°", fontsize=9)
        ax.axis("equal")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def test_straight_loss():
    def create_triangle(angle_deg, length=1.0):
        """
        Create 3 points (A, B, C) with angle at B = angle_deg.
        - angle=180 -> straight line
        - angle=0   -> folded
        """
        angle_rad = math.radians(angle_deg)

        # B is the vertex
        B = torch.tensor([0.0, 0.0])
        # A fixed to the left
        A = torch.tensor([-length, 0.0])
        # BA vector
        BA = A - B

        # Rotate BA by angle_deg to get BC
        rot = torch.tensor([
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad),  math.cos(angle_rad)]
        ])
        BC = torch.matmul(rot, BA)

        C = B + BC
        return torch.stack([A, B, C], dim=0)

    # Prepare data
    angles = range(0, 181)
    losses = []
    for ang in angles:
        pts = create_triangle(ang).float()
        loss = StraightnessLoss().forward(pts, is_close=False)
        # loss = cal_straightness_loss_v1(pts, is_close=False)
        losses.append(loss.item())

    # --- Create ONE big figure ---
    fig = plt.figure(figsize=(18, 10))

    # --- First subplot: loss vs angle ---
    ax1 = plt.subplot2grid((3, 7), (0, 0), colspan=7)  # top row, full width
    ax1.plot(angles, losses, marker="o")
    # ax1.set_ylim(0, 1)
    ax1.set_xlabel("Angle (degrees)")
    ax1.set_ylabel("Loss")
    ax1.set_title("Straightness loss vs angle")
    ax1.grid(True)
    ax1.set_xticks(range(0, 181, 15))

    # --- Second subplot: polyline grid ---
    grid_angles = range(0, 181, 15)  # 0, 15, 30, ..., 180
    for i, ang in enumerate(grid_angles):
        row, col = divmod(i, 7)
        ax = plt.subplot2grid((3, 7), (row + 1, col))  # rows 1–2
        pts = create_triangle(ang).float()
        x, y = pts[:, 0], pts[:, 1]
        ax.plot(x, y, "o-", linewidth=2)
        ax.set_title(f"{ang}°", fontsize=10)
        ax.axis("equal")
        ax.axis("off")

    plt.suptitle("Straightness Loss + Polylines at Different Angles", fontsize=16)
    plt.tight_layout()
    plt.show()


def test_axis_align_loss():
    angles_deg = np.arange(0, 361, 1)  # full sweep
    losses = []

    for ang in angles_deg:
        rad = math.radians(ang)
        p0 = torch.tensor([0.0, 0.0])
        p1 = torch.tensor([math.cos(rad), math.sin(rad)])
        pts = torch.stack([p0, p1], dim=0)

        loss = cal_axis_align_loss(pts)
        losses.append(loss.item())

    # --- Figure 1: loss curve ---
    plt.figure(figsize=(12, 6))
    plt.plot(angles_deg, losses, marker='o', markersize=3)
    plt.xlabel("Edge angle (degrees)")
    plt.ylabel("Axis alignment loss")
    plt.title("Axis alignment loss vs edge angle")

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(45))
    ax.set_ylim(0, 1.0)
    plt.grid(True)

    # --- Figure 2: polyline samples ---
    plt.figure(figsize=(12, 12))
    angles_show = range(0, 360, 15)

    for i, ang in enumerate(angles_show, 1):
        rad = math.radians(ang)
        p0 = torch.tensor([0.0, 0.0])
        p1 = torch.tensor([math.cos(rad), math.sin(rad)])
        pts = torch.stack([p0, p1], dim=0)

        x, y = pts[:, 0], pts[:, 1]

        plt.subplot(4, 6, i)  # 25 subplots max → fits in 30 slots
        plt.plot(x, y, "o-", linewidth=2)
        plt.title(f"{ang}°", fontsize=9)
        plt.axis("equal")
        plt.axis("off")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    test_straight_loss()