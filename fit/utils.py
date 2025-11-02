import torch
import numpy as np
from .share import sh

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from pathlib import Path

# def visualize_pts(points, png_name,
#                   canvas_size=320,
#                   stroke="blue", stroke_width=2,
#                   point_radius=2, point_color="red",
#                   first_point_color="yellow",
#                   close_path=True,
#                   background_image=None,
#                   midpoint_indices=None,
#                   point_values=None,
#                   binary_color=False,
#                   show_indices=True,             # <--- NEW ARG
#                   index_color="white",           # <--- NEW ARG
#                   index_font_size=10):           # <--- NEW ARG
#     """
#     Draw [N,2] points directly into a raster PNG with optional background image.
#     Points should be normalized [0,1].
#     background_image: np.ndarray (H,W,3) in uint8 or float [0,1].
#     point_values: array-like of numbers to map to colors (blue to red)
#     """

#     # Convert torch -> numpy
#     if isinstance(points, torch.Tensor):
#         points = points.detach().cpu().numpy()

#     pts = np.asarray(points, dtype=float)
#     if pts.ndim != 2 or pts.shape[1] != 2:
#         raise ValueError(f"Expected shape [N,2], got {pts.shape}")

#     draw_pts = pts * canvas_size
#     if close_path:
#         draw_pts = np.vstack([draw_pts, draw_pts[0]])

#     # Background
#     if background_image is not None:
#         bg = np.asarray(background_image)
#         if bg.shape[2] == 4:
#             bg = bg[..., :3]
#         h, w, _ = bg.shape
#         scale = canvas_size // h
#         bg_up = np.repeat(np.repeat(bg, scale, axis=0), scale, axis=1)
#         if bg_up.dtype != np.uint8:
#             bg_up = (bg_up * 255).astype(np.uint8)
#         img = Image.fromarray(bg_up)
#     else:
#         img = Image.new("RGB", (canvas_size, canvas_size), "black")

#     draw = ImageDraw.Draw(img)

#     # Draw polyline
#     draw.line([tuple(p) for p in draw_pts], fill=stroke, width=stroke_width)

#     iterable_pts = draw_pts[:-1] if close_path else draw_pts

#     # Value-based coloring
#     colors = None
#     if point_values is not None:
#         point_values = np.array(point_values)
#         if len(point_values) != len(iterable_pts):
#             raise ValueError(f"point_values length {len(point_values)} must match points length {len(iterable_pts)}")
#         min_val, max_val = np.min(point_values), np.max(point_values)
#         normalized_values = (point_values - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(point_values)
#         colors = []
#         if binary_color:
#             for v in point_values:
#                 colors.append("rgb(0,255,0)" if v else "rgb(255,0,0)")
#         else:
#             for val in normalized_values:
#                 r = int(255 * val)
#                 b = int(255 * (1 - val))
#                 colors.append(f"rgb({r},0,{b})")

#     # Font for indices
#     font = None
#     if show_indices:
#         try:
#             font = ImageFont.truetype("arial.ttf", index_font_size)
#         except:
#             font = ImageFont.load_default()

#     # Draw points and optionally indices
#     for i, (x, y) in enumerate(iterable_pts):
#         color = (
#             colors[i] if colors is not None else
#             first_point_color if i == 0 else
#             point_color
#         )
#         draw.ellipse(
#             [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
#             fill=color
#         )

#         # 🆕 Draw index label
#         if show_indices:
#             offset = 4  # small gap between point and label
#             draw.text((x + point_radius + offset, y - point_radius - offset),
#                       str(i), fill=index_color, font=font)

#     png_name = Path(png_name)
#     png_name.parent.mkdir(parents=True, exist_ok=True)
#     img.save(png_name)


def visualize_matrix(matrix, save_path, **kwargs):
    """Visualize a 2D matrix as a heatmap and save as PNG.

    Args:
        matrix (np.ndarray or torch.Tensor): 2D array to visualize (n x n or H x W).
        save_path (str or Path): path to save the PNG file.
    
    Keyword Args:
        cmap (str): colormap name (default 'viridis')
        vmin, vmax (float): color limits (default: auto)
        annotate (bool): overlay numeric values (default False)
        fmt (str): format string for annotations (default '.2f')
        figsize (tuple): figure size (default (6, 6))
        dpi (int): output DPI (default 150)
        colorbar (bool): show colorbar (default True)

    Example:
        visualize_matrix(mat, 'out/heatmap.png', annotate=True, cmap='plasma')
    """
    # We import plotting libraries here so this function can be defined early in the file
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Extract parameters with defaults
    cmap = kwargs.get('cmap', 'viridis')
    # vmin = kwargs.get('vmin', None)
    vmin = matrix.min()
    vmax = matrix.max()
    annotate = kwargs.get('annotate', False)
    fmt = kwargs.get('fmt', '.2f')
    figsize = kwargs.get('figsize', (6, 6))
    dpi = kwargs.get('dpi', 150)
    colorbar = kwargs.get('colorbar', True)

    # Convert to numpy
    if isinstance(matrix, torch.Tensor):
        mat = matrix.detach().cpu().numpy()
    else:
        mat = np.asarray(matrix)

    if mat.ndim != 2:
        raise ValueError(f"visualize_matrix expects a 2D matrix, got shape {mat.shape}")

    # Create output directory
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal', origin='upper')

    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([])
    ax.set_yticks([])

    if annotate:
        # annotate each cell with its value
        nrows, ncols = mat.shape
        for i in range(nrows):
            for j in range(ncols):
                text = format(mat[i, j], fmt)
                ax.text(j, i, text, ha='center', va='center', color='white', fontsize=6)

    plt.tight_layout()
    
    fig.savefig(str(save_path), dpi=dpi)
    plt.close(fig)
    return str(save_path)

def grad_to_pt(img_grad, smooth_grad, band_grad, straight_grad, pt_name):
    """
    Save gradients as a .pt file.
    Each gradient is an [N,2] tensor.
    """

    grads = torch.stack([img_grad.detach().cpu(), smooth_grad.detach().cpu(), band_grad.detach().cpu(), straight_grad.detach().cpu()])
    grads = grads.permute(1, 0, 2)

    # Ensure parent directory exists
    pt_path = Path(pt_name)
    pt_path.parent.mkdir(parents=True, exist_ok=True)

    # Write pt file
    torch.save(grads, pt_path)

def visualize_angles(points, is_close, save_path, eps=1e-8):
    if is_close:
        points = torch.cat([points, points[:2]], dim=0)

    n = points.shape[0]
    if n < 3:
        return

    ba = points[:-2] - points[1:-1]
    bc = points[2:]  - points[1:-1]

    ba = ba / (ba.norm(dim=1, keepdim=True) + eps)
    bc = bc / (bc.norm(dim=1, keepdim=True) + eps)

    cos_theta = (ba * bc).sum(dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    angles = torch.acos(cos_theta) * 180.0 / torch.pi  # in degrees

    angles = torch.flip(angles, dims=[0]).cpu().numpy()

    plt.figure(figsize=(6, 4))
    plt.plot(angles, marker='o')
    plt.scatter(0, angles[0], color='green', s=80, zorder=5, label="First point, go clockwise")

    plt.xlabel("Index of angle (triplet)")
    plt.ylabel("Angle (degrees)")
    plt.title("Angles between consecutive segments")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def sample_circle(center, radius, n):
    """
    Sample n evenly spaced points along the circumference of a circle.

    Args:
        center (tuple): (cx, cy) of the circle center (constant).
        radius (float): circle radius (constant).
        n (int): number of sample points.

    Returns:
        torch.Tensor: [n, 2] tensor of sampled points (requires_grad=True).
    """
    cx, cy = center

    # Evenly spaced angles (no endpoint=True needed)
    angles = torch.arange(0, n, dtype=torch.float32) * (2 * torch.pi / n)

    # Circle coordinates
    x = cx + radius * torch.cos(angles)
    y = cy + radius * torch.sin(angles)

    points = torch.stack([x, y], dim=1)

    return points.requires_grad_()

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ipdb
from pathlib import Path

def points_to_pt(points, pt_name):
    """
    Save [N,2] points as a .pt file.
    Points are written at their given coordinates without normalization.
    """

    pts = np.asarray(points.detach().numpy(), dtype=float)

    # Ensure parent directory exists
    pt_path = Path(pt_name)
    pt_path.parent.mkdir(parents=True, exist_ok=True)

    # Write pt file
    torch.save(torch.tensor(pts, dtype=torch.float32), pt_path)

def points_to_png(points, png_name,
                  canvas_size=320,
                  stroke="blue", stroke_width=2,
                  point_radius=2, point_color="red",
                  first_point_color="yellow",
                  close_path=True,
                  background_image=None,
                  midpoint_indices=None,
                  point_values=None,
                  binary_color = False):
    """
    Draw [N,2] points directly into a raster PNG with optional background image.
    Points should be normalized [0,1].
    background_image: np.ndarray (H,W,3) in uint8 or float [0,1].
    point_values: array-like of numbers to map to colors (blue to red)
    """

    # Convert torch -> numpy
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected shape [N,2], got {pts.shape}")

    # Scale to pixel coords
    # draw_pts = (pts + 0.5) * canvas_size 

    draw_pts = pts * canvas_size

    if close_path:
        draw_pts = np.vstack([draw_pts, draw_pts[0]])


    # Use sh.bg_color for background if no image
    # ipdb.set_trace()
    if background_image is not None:
        bg = np.asarray(background_image)

        if bg.shape[2] == 4:
            bg = bg[..., :3]  # discard alpha channel

        if bg.ndim != 3 or bg.shape[2] != 3:
            raise ValueError("background_image must be (H, W, 3)")

        h, w, _ = bg.shape
        if canvas_size % h != 0 or canvas_size % w != 0:
            raise ValueError("canvas_size must be integer multiple of background size")

        scale = canvas_size // h  # assumes square bg
        # Nearest neighbor upscale
        bg_up = np.repeat(np.repeat(bg, scale, axis=0), scale, axis=1)

        if bg_up.dtype != np.uint8:
            if bg_up.max() <= 1.0:
                bg_up = (bg_up * 255).astype(np.uint8)
            else:
                bg_up = bg_up.astype(np.uint8)

        img = Image.fromarray(bg_up)
    else:
        img = Image.new("RGB", (canvas_size, canvas_size), sh.bg_color)

    draw = ImageDraw.Draw(img)

    # Use sh.outline_color for stroke, sh.point_color for points, sh.contour_color for first point, sh.point_color for highlight
    stroke = sh.outline_color_hex if hasattr(sh, 'outline_color_hex') else sh.outline_color
    point_color = sh.point_color_hex if hasattr(sh, 'point_color_hex') else sh.point_color
    first_point_color = sh.first_point_color if hasattr(sh, 'first_point_color') else sh.first_point_color
    highlight_color = sh.point_color_hex if hasattr(sh, 'point_color_hex') else sh.point_color

    # Draw polyline
    draw.line([tuple(p) for p in draw_pts], fill=stroke, width=stroke_width)

    # Draw points
    iterable_pts = draw_pts[:-1] if close_path else draw_pts
    
    # Prepare color mapping if point_values are provided
    colors = None
    if point_values is not None:
        point_values = np.array(point_values)
        if len(point_values) != len(iterable_pts):
            raise ValueError(f"point_values length {len(point_values)} must match points length {len(iterable_pts)}")
        
        # Normalize values to [0, 1]
        min_val, max_val = np.min(point_values), np.max(point_values)
        if max_val > min_val:
            normalized_values = (point_values - min_val) / (max_val - min_val)
        else:
            normalized_values = np.zeros_like(point_values)
        
        # Map from blue (0) to red (1)
        colors = []
        if binary_color:
            for i in range(len(iterable_pts)):
                if point_values[i] == 0:
                    colors.append("rgb(255,0,0)")  # Red
                else:
                    colors.append("rgb(0,255,0)")  # Green
        else:
            for val in normalized_values:
            # Blue to red: (255*val, 0, 255*(1-val))
                r = int(255 * val)
                g = 0
                b = 0
                colors.append(f"rgb({r},{g},{b})")
    
    for i, (x, y) in enumerate(iterable_pts):
        # Use value-based color if provided
        if colors is not None:
            color = colors[i]
        # Use highlight color for midpoints if specified
        elif midpoint_indices is not None and i in midpoint_indices:
            color = highlight_color
        elif i == 0:
            color = first_point_color
        else:
            color = point_color
        draw.ellipse(
            [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
            fill=color
        )

    png_name = Path(png_name)
    png_name.parent.mkdir(parents=True, exist_ok=True)
    img.save(png_name)




def points_to_svg(points, svg_name,
                  stroke="blue", stroke_width=1,
                  point_radius=0.6, point_color="red",
                  close_path=True):
    """
    Save [N,2] points as an SVG polyline with optional circles at points.
    Points are                  background_image=None,out normalization.
    """

    from pathlib import Path
    import numpy as np

    try:
        import torch
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
    except ImportError:
        pass

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected shape [N,2], got {pts.shape}")

    draw_pts = pts * sh.w
    if close_path:
        draw_pts = np.vstack([draw_pts, draw_pts[0]])

    # Polyline string
    polyline_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in draw_pts)

    # Build SVG
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{sh.w}" height="{sh.w}">']
    svg.append(f'  <polyline points="{polyline_str}" fill="none" stroke="{stroke}" stroke-width="{stroke_width}"/>')
    for x, y in draw_pts[:-1] if close_path else draw_pts:
        svg.append(f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="{point_radius}" fill="{point_color}"/>')
    svg.append('</svg>')

    svg_str = "\n".join(svg)

    # Ensure parent directory exists
    svg_path = Path(svg_name)
    svg_path.parent.mkdir(parents=True, exist_ok=True)

    # Write SVG file
    with svg_path.open("w") as f:
        f.write(svg_str)


def add_midpoints(points: torch.Tensor, closed: bool = True) -> torch.Tensor:
    # same implementation you had
    midpoints = (points[:-1] + points[1:]) / 2
    if closed:
        closing_mid = (points[-1] + points[0]) / 2
        midpoints = torch.cat([midpoints, closing_mid.unsqueeze(0)], dim=0)
    return torch.stack([points, midpoints], dim=1).reshape(-1, 2)

def expand_points_param(points_param: torch.nn.Parameter,
                        optimizer_cls,
                        lr=1e-2,
                        closed=True):
    """
    Double the number of points by adding midpoints, and return a new nn.Parameter
    plus a fresh optimizer.

    Args:
        points_param (nn.Parameter): Current trainable points, shape (N,2).
        optimizer_cls: Optimizer class (e.g. torch.optim.Adam).
        lr (float): Learning rate for new optimizer.
        closed (bool): Whether to close the loop when adding midpoints.

    Returns:
        new_points_param (nn.Parameter), new_optimizer
    """
    with torch.no_grad():
        expanded = add_midpoints(points_param.data, closed=closed)  # break graph

    new_points_param = torch.nn.Parameter(expanded)
    new_optimizer = optimizer_cls([new_points_param], lr=lr)
    return new_points_param, new_optimizer


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

# def detect_redundant_edge(points, contour):
#     def test_redundant_edge(p1, p2, mask):
#         def horizontal_test():
#             x = int(p1[0])
#             y_start = int(min(p1[1], p2[1]))
#             y_end = int(max(p1[1], p2[1]))
#             for y in range(y_start, y_end + 1):
#                 if mask[x, y]:
#                     if x + 1 < sh.w and mask[x + 1, y]:
#                         return False
#                     if x - 1 >= 0 and mask[x - 1, y]:
#                         return False
#                 else:return False
#             return True

#         def vertical_test():
#             y = int(p1[1])
#             x_start = int(min(p1[0], p2[0]))
#             x_end = int(max(p1[0], p2[0]))
#             for x in range(x_start, x_end + 1):
#                 if mask[x, y]:
#                     if y + 1 < sh.w and mask[x, y + 1]:
#                         return False
#                     if y - 1 >= 0 and mask[x, y - 1]:
#                         return False
#                 else:return False
#             return True

#         if p1[0] == p2[0]:
#             # ipdb.set_trace()
#             return horizontal_test()
#         elif p1[1] == p2[1]:
#             # ipdb.set_trace()
#             return vertical_test()
#         else:
#             return False


#     mask = torch.zeros((sh.w, sh.w), dtype=torch.bool, device=contour.device)
#     xy = np.floor(contour - 0.5).astype(np.int64)

#     # filter valid coordinates
#     valid = (xy[:, 0] >= 0) & (xy[:, 0] < sh.w) & (xy[:, 1] >= 0) & (xy[:, 1] < sh.w)
#     xy = xy[valid]

#     # mark in mask
#     mask[xy[:, 0], xy[:, 1]] = True

#     points = torch.cat([points, points[:1]], dim=0)
#     redundant_edge = []
#     for i in range(len(points)):
#         p1 = (points[i]  *sh.w).floor()
#         p2 = (points[(i+1)%len(points)] *sh.w).floor()
#         if test_redundant_edge(p1, p2, mask):
#             redundant_edge.append(i)
#     return redundant_edge

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