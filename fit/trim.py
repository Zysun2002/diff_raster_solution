import torch
import numpy as np
from .share import sh
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pydiffvg
from .loss import SmoothnessLoss, BandLoss, ImageLoss, StraightnessLoss
import ipdb
from pathlib import Path
import random
from .utils import points_to_png

def is_need_more_points(point_n, contour_img):

    def polygon_to_contour_mask(points: torch.Tensor, resolution: int, width: int = 1) -> torch.Tensor:
        """
        Rasterize only the contour of a closed polygon into a binary mask.

        Args:
            points (torch.Tensor): shape (N,2), polygon vertices in [0,1] range.
            resolution (int): output mask resolution (mask will be resolution x resolution).
            width (int): line thickness in pixels.

        Returns:
            torch.Tensor: binary mask (H,W), dtype=torch.bool
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must be (N,2)")

        # scale to pixel coordinates
        pts = (points.detach().cpu().numpy() * resolution).astype(np.int32)

        # blank image
        mask_img = Image.new("L", (resolution, resolution), 0)
        draw = ImageDraw.Draw(mask_img)

        # draw only the polygon outline (no fill)
        draw.line([tuple(p) for p in pts] + [tuple(pts[0])], fill=1, width=width)

        # convert to torch mask
        mask = torch.from_numpy(np.array(mask_img, dtype=np.uint8))
        return mask.bool()


    vec_mask = polygon_to_contour_mask(point_n, sh.w)

    ct_mask_np = (contour_img == 255).all(axis=-1)
    ct_mask = torch.from_numpy(ct_mask_np)

    overlap_count = (vec_mask * ct_mask).sum().item()
    # ipdb.set_trace()

    print("ratio:", overlap_count / vec_mask.sum())

def detect_outlier_edge(points, contour):

    
    def line_within_boundary(p1, p2, contour):
        from skimage.draw import line

        boundary_set = {tuple(map(int, pt)) for pt in contour}
        
        # Get pixel coordinates for the line
        rr, cc = line(int(p1[1]), int(p1[0]), int(p2[1]), int(p2[0]))  # note y,x order
        line_pixels = set(zip(cc, rr))
        
        # Check if all line pixels are within the boundary
        return line_pixels.issubset(boundary_set)

    torch.cat([points, points[:2]], dim=0)
    outlier_edge = []
    for i in range(len(points)):
        p1 = points[i]  *sh.w
        p2 = points[(i+1)%len(points)] *sh.w
        if not line_within_boundary(p1, p2, contour):
            outlier_edge.append(i)
    return outlier_edge


def insert_midpoints(points_n, outlier_edge):
    N = points_n.shape[0]
    outlier_edge = torch.as_tensor(outlier_edge, device=points_n.device, dtype=torch.long)
    
    new_points = []
    midpoint_indices = []
    for i in range(N):
        new_points.append(points_n[i])
        if (outlier_edge == i).any():  # check if edge i→i+1 is outlier
            j = (i + 1) % N
            midpoint = (points_n[i] + points_n[j]) / 2.0
            new_points.append(midpoint)
            midpoint_indices.append(len(new_points) - 1)
    
    return torch.stack(new_points, dim=0), midpoint_indices


def insert_closest_band_point(points_n, outlier_edge, contour):
    def find_closest_point(point, contour):
        # Ensure point and contour are tensors

        contour = torch.tensor(contour, device=contour.device, dtype=point.dtype) / sh.w
        closest_point = None
        min_dist = torch.tensor(float("inf"), device=point.device)
        for cp in contour:
            dist = torch.norm(point - cp)
            if dist < min_dist:
                min_dist = dist
                closest_point = cp

        return closest_point

    N = points_n.shape[0]
    outlier_edge = torch.as_tensor(outlier_edge, device=points_n.device, dtype=torch.long)

    new_points = []
    added_point_indices = []
    for i in range(points_n.shape[0]):
        new_points.append(points_n[i])
        if (outlier_edge == i).any():  # check if edge i→i+1 is outlier
            closest_point = find_closest_point((points_n[i] + points_n[(i + 1) % N]) / 2.0, contour)
            new_points.append(closest_point)
            added_point_indices.append(len(new_points) - 1)
    return torch.stack(new_points, dim=0), added_point_indices


def detect_redundant_point_by_edge(points):
    # if two edges are sufficiently close, consider one redundant
    redundant_point = []

    points = torch.cat([points, points[:2]], dim=0)
    cos = torch.cosine_similarity(points[1:-1] - points[:-2], points[1:-1] - points[2:], dim=1)
    threshold = -0.99  # cos(8 degrees) ~ 0.98
    # ipdb.set_trace()
    for i in range(len(cos)):
        if cos[i] < threshold:
            redundant_point.append((i + 1) % len(points))

    return redundant_point

def remove_redundant_edge(points_n, redundant_edges):
    N = points_n.shape[0]
    redundant_edges = torch.as_tensor(redundant_edges, device=points_n.device, dtype=torch.long)
    
    remove_indices = torch.tensor([(i+1)%N for i in range(N) if i in redundant_edges and (i+1)%N in redundant_edges])

    # Create mask for all points to keep
    if remove_indices.numel() == 0:
        return points_n
    keep_mask = torch.ones(N, dtype=torch.bool, device=points_n.device)
    keep_mask[remove_indices] = False

    # Apply mask
    new_points = points_n[keep_mask]

    return new_points

def remove_redundant_point(points_n, redundant_points):
    N = points_n.shape[0]
    redundant_points = torch.as_tensor(redundant_points, device=points_n.device, dtype=torch.long)

    remove_indices = torch.tensor([i for i in range(N) if i in redundant_points])
    # Create mask for all points to keep
    # remove_indices = redundant_points
    if remove_indices.numel() == 0:
        return points_n
    keep_mask = torch.ones(N, dtype=torch.bool, device=points_n.device)
    keep_mask[remove_indices] = False

    # Apply mask
    new_points = points_n[keep_mask]

    return new_points


def find_midpoint_on_contour(contour_mask, start, end):
    """
    contour_mask: torch.BoolTensor or np.ndarray, shape [H, W], True for contour pixels
    start, end: (x, y) pixel coordinates (ints)
    Returns: (x, y) pixel coordinate of midpoint along shortest contour path
    """
    from collections import deque

    H, W = contour_mask.shape
    visited = np.zeros((H, W), dtype=bool)
    prev = np.full((H, W, 2), -1, dtype=int)
    queue = deque()
    queue.append(start)
    visited[start[1], start[0]] = True

    while queue:
        x, y = queue.popleft()
        if (x, y) == end:
            break
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < W and 0 <= ny < H and contour_mask[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                prev[ny, nx] = [x, y]
                queue.append((nx, ny))

    # Reconstruct path
    path = []
    x, y = end
    while (x, y) != start and prev[y, x][0] != -1:
        path.append((x, y))
        x, y = prev[y, x]
    path.append(start)
    path = path[::-1]

    if len(path) == 0:
        return None  # No path found

    mid_idx = len(path) // 2
    return path[mid_idx]


def detect_redundant_point(points):
    # if two points are sufficiently close, consider one redundant
    redundant_point = []
    threshold = 2.0 / sh.w  # threshold in normalized coords
    N = points.shape[0]                 
    for i in range(N):
        p1 = points[i]
        p2 = points[(i + 1) % N]
        if torch.norm(p1 - p2) < threshold:
            redundant_point.append((i + 1) % N)
    return redundant_point

def trim(points_n):
    # previous trim
    redundant_point = detect_redundant_point(points_n)
    points_n = remove_redundant_point(points_n, redundant_point)

    redundant_point = detect_redundant_point_by_edge(points_n)
    points_n = remove_redundant_point(points_n, redundant_point)
    
    return points_n.detach().clone().requires_grad_(True)

def remove_points_based_on_loss(points_n, raster):
    render = pydiffvg.RenderFunction.apply


    color_n = torch.tensor(sh.color_guess, requires_grad=True)
    polygon = pydiffvg.Polygon(points = points_n, is_closed = True)
    shapes = [polygon]
    polygon_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                        fill_color = color_n)
    shape_groups = [polygon_group]
    polygon.points = points_n * sh.w
    polygon_group.color = color_n

    shapes[0].points = points_n * sh.w

    polygon_group.fill_color = color_n
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        sh.w, sh.w, shapes, shape_groups)
    

    seed = random.randint(0, 2**31 - 1)
    img_prev = render(sh.w,   # width
            sh.w,   # height
            sh.num_mc_x,     # num_samples_x
            sh.num_mc_y,     # num_samples_y
            seed + 117,   # seed
            sh.background, # background_image
            *scene_args)
    
    loss_prev = ImageLoss()(img_prev, raster)
    print(loss_prev.item())

    # i want to test each point redundancy by removing it and checking loss increase
    N = points_n.shape[0]
    points_n_detached = points_n.detach().clone()
    points_to_remove = [] 
    points_loss = []      

    for i in range(N):  
        test_points = torch.cat([points_n_detached[:i], points_n_detached[i+1:]], dim=0)
        polygon.points = test_points * sh.w
        shapes[0].points = test_points * sh.w

        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            sh.w, sh.w, shapes, shape_groups)

        seed = random.randint(0, 2**31 - 1)
        img_test = render(sh.w,   # width
                sh.w,   # height
                sh.num_mc_x,     # num_samples_x
                sh.num_mc_y,     # num_samples_y
                seed + 117,   # seed
                sh.background, # background_image
                *scene_args)
        
        loss_test = ImageLoss()(img_test, raster)
        loss_diff = loss_test - loss_prev
        print(f"Point {i}, diff loss: {loss_diff.item()}")
        # if loss_test - loss_prev < 0.01:  # threshold for loss increase
        #     points_to_remove.append(i)
        points_loss.append(loss_diff.item())

    # thres = 0.03
    # points_loss = np.array([1 if points_loss[i]> thres else 0 for i in range(len(points_loss))])

    return points_loss
    # ipdb.set_trace()

def add_points_based_on_optimization(points_n, add_points_sh):
    render = pydiffvg.RenderFunction.apply

    for i in range(len(points_n)-1):
        mid_point = (points_n[i] + points_n[i+1]) / 2.0
        mid_point = mid_point.detach().requires_grad_(True)  # Only the midpoint requires grad
        # ipdb.set_trace()
        # Create points with frozen original points and trainable midpoint
        points_before = points_n[:i+1].detach()  # Frozen
        points_after = points_n[i+1:].detach()   # Frozen
        points_with_mid = torch.cat([points_before, mid_point.unsqueeze(0), points_after], dim=0)
        points_init = points_with_mid.detach().clone()

        color_n = torch.tensor(sh.color_guess, requires_grad=True)
        polygon = pydiffvg.Polygon(points = points_with_mid, is_closed = True)
        shapes = [polygon]
        polygon_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                            fill_color = color_n)
        shape_groups = [polygon_group]
        polygon.points = points_with_mid * sh.w
        polygon_group.color = color_n

        optimizer = torch.optim.Adam([mid_point], lr=1e-3)  # Only optimize the midpoint

        for t in range(add_points_sh.epoch):
        
            optimizer.zero_grad()
            # Reconstruct points_with_mid with updated midpoint
            points_with_mid = torch.cat([points_before, mid_point.unsqueeze(0), points_after], dim=0)
            
            # Forward pass: render the image.
            shapes[0].points = points_with_mid * sh.w

            polygon_group.fill_color = color_n
            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                sh.w, sh.w, shapes, shape_groups)
            
            seed = random.randint(0, 2**31 - 1)

            img = render(sh.w,   # width
                        sh.w,   # height
                        sh.num_mc_x,     # num_samples_x
                        sh.num_mc_y,     # num_samples_y
                        seed + 117,   # seed
                        sh.background, # background_image
                        *scene_args)
            
            img_loss = 10 * ImageLoss()(img, sh.raster)
            # img_loss = 2 * (img - raster).pow(2).mean() 

            # smooth_loss = SmoothnessLoss()(points_n, points_init=points_init, is_close=True)
            
            smooth_loss = SmoothnessLoss(add_points_sh.smooth_loss)(points_with_mid, points_init=points_init, is_close=True)

            band_loss = BandLoss(add_points_sh.band_loss)(points_with_mid, sh.udf)

            # straightness_loss = StraightnessLoss(add_points_sh.straightness_loss)(points_with_mid, is_close=True)

            loss = img_loss

            # ipdb.set_trace()
            loss.backward()
            optimizer.step()

            # ipdb.set_trace()

        point_values = np.zeros(points_with_mid.shape[0])
        point_values[i+1] = 1
        points_to_png(
            points_with_mid.detach().cpu().numpy(), 
            sh.sub_path / "add_points" / f"add_points_iter_{i:02}.png",
            background_image=sh.raster,
            point_values=point_values
        )

    return points_n.detach()

        # ipdb.set_trace()
