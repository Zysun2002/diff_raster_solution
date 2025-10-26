import numpy as np
import cv2
from PIL import Image
import torch
import ipdb
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from .share import sh

import numpy as np

def fill_hole_in_contour(contour):
    contour = contour.astype(int)
    N = contour.shape[0]
    new_points = []

    for i in range(N):
        p1 = contour[i]
        p2 = contour[(i + 1) % N]  # wrap around for closed contour
        new_points.append(p1.tolist())

        dy, dx = (p2 - p1).tolist()

        # If diagonal (8-connected)
        if abs(dx) == 1 and abs(dy) == 1:
            # Add both possible 4-neighbor pixels
            new_points.append([p1[0], p1[1] + dx])  # horizontal neighbor
            new_points.append([p1[0] + dy, p1[1]])  # vertical neighbor

        # If distance > 1 pixel (fill in between)
        elif abs(dx) > 1 or abs(dy) > 1:
            steps = max(abs(dx), abs(dy))
            for s in range(1, steps):
                new_points.append([
                    int(round(p1[0] + s * dy / steps)),
                    int(round(p1[1] + s * dx / steps))
                ])

    return np.array(new_points, dtype=int)

        
def extract_4_connected_contour(img):
    h, w = img.shape

    # Directions: right, down, left, up (clockwise)
    dirs = [(0,1),(1,0),(0,-1),(-1,0)]

    # find starting pixel (top-left-most foreground with a background 4-neighbor)
    start = None
    for i in range(h):
        for j in range(w):
            if img[i,j] == 1:
                for di,dj in dirs:
                    ni, nj = i+di, j+dj
                    if ni<0 or ni>=h or nj<0 or nj>=w or img[ni,nj]==0:
                        start = (i,j)
                        break
            if start is not None:
                break
        if start is not None:
            break

    if start is None:
        return np.empty((2,0), dtype=int)

    contour = [start]
    curr = start
    prev_dir = 3  # initial direction "up" (arbitrary)
    
    while True:
        found = False
        # check 4 neighbors in clockwise order starting from prev_dir
        for k in range(4):
            dir_idx = (prev_dir + k) % 4
            di, dj = dirs[dir_idx]
            ni, nj = curr[0]+di, curr[1]+dj
            if 0<=ni<h and 0<=nj<w and img[ni,nj]==1:
                if (ni,nj) != contour[-1]:  # avoid immediate backtracking
                    contour.append((ni,nj))
                    curr = (ni,nj)
                    prev_dir = (dir_idx + 3) % 4  # turn left as next search start
                    found = True
                    break
        if not found or curr == start:
            break

    contour = np.array(contour, dtype=int)
    
    return contour[:, [1, 0]]


def extract_sdf_from_contour(contour_image, sdf_path):
    # Ensure grayscale
    if contour_image.ndim == 3:
        gray = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = contour_image

    # Binarize (contour = 1, background = 0)
    _, contour = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)

    # computes the Euclidean distance to the nearest zero value
    D = distance_transform_edt(1 - contour)

    # Normalize for visualization (0-255)
    D_norm = (D - D.min()) / (D.max() - D.min())
    D_img = (D_norm * 255).astype(np.uint8)

    # Convert to color (BGR)
    D_color = cv2.cvtColor(D_img, cv2.COLOR_GRAY2BGR)

    # Mark contour pixels (D == 0) as red
    D_color[D == 0] = [0, 0, 255]  # BGR red

    # Save image
    cv2.imwrite(sdf_path, D_color)

    ret = torch.tensor(D / sh.w, dtype=torch.float32)

    return ret


def sample_from_boundary(image_path, contour_path=None):
    """
    Extract evenly spaced boundary points from an image containing
    a single closed object, normalized to [0,1].

    Args:
        image_path (str): path to the raster image
        num_points (int): number of points to sample along the boundary

    Returns:
        torch.nn.Parameter: (num_points, 2) normalized (x,y) points
                            with requires_grad=True
    """
    # === Step 1: Load and grayscale ===
    img = np.array(Image.open(image_path).convert("L"))
    h, w = img.shape  # height, width
    
    # === Step 2: Binarize ===
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # ipdb.set_trace()

    cv2.imwrite(image_path.with_name('binary.png'), binary)
    
    

    # === Step 3: Find contour ===
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found in the image")
    contour = max(contours, key=cv2.contourArea).squeeze()  # (N, 2)

    contour = fill_hole_in_contour(contour)

    if contour_path: 
        # create a blank background using sh.bg_color
        contour_img = np.full((h, w, 3), sh.bg_color, dtype=np.uint8)
        # draw contour using sh.contour_color (thickness = 2 px)
        cv2.drawContours(contour_img, [contour.reshape(-1, 1, 2)], -1, sh.contour_color, 1)
        # save as PNG
        cv2.imwrite(contour_path, contour_img)

    contour_img_for_udf = np.zeros((h, w, 3), dtype=np.uint8)
            # draw contour in white (thickness = 2 px)
    cv2.drawContours(contour_img_for_udf, [contour.reshape(-1, 1, 2)], -1, (255, 255, 255), 1)

    contour = contour + 0.5

    udf = extract_sdf_from_contour(contour_img_for_udf, contour_path.with_name("sdf.png"))


    # === Step 5: Sample evenly spaced points ===

    sample_points = InitializationSampler()(contour)

    

    # === Step 7: Return as nn.Parameter ===
    return  sample_points, udf, contour_img, contour

class InitializationSampler():
    def __init__(self):
        pass

    def __call__(self, contour):
        return self.forward0(contour)
    
    def forward0(self, contour):

        diffs = np.diff(contour, axis=0, append=contour[:1])
        segment_lengths = np.sqrt((diffs**2).sum(axis=1))
        cumlen = np.cumsum(segment_lengths)
        cumlen = np.insert(cumlen, 0, 0)
        total_len = cumlen[-1]

        target_lens = np.linspace(0, total_len, sh.num_samples, endpoint=False)
        sampled_points = []
        for t in target_lens:
            idx = np.searchsorted(cumlen, t) - 1
            idx = np.clip(idx, 0, len(contour) - 1)

            seg_start, seg_end = contour[idx], contour[(idx + 1) % len(contour)]
            seg_len = segment_lengths[idx]

            if seg_len == 0:
                sampled_points.append(seg_start)
            else:
                alpha = (t - cumlen[idx]) / seg_len
                pt = (1 - alpha) * seg_start + alpha * seg_end
                sampled_points.append(pt)

        sampled_points = np.array(sampled_points, dtype=np.float32)  # (num_points, 2)

        # === Step 6: Normalize to [0,1] ===
        sampled_points[:, 0] /= sh.w   # normalize x by width
        sampled_points[:, 1] /= sh.w   # normalize y by height

        return torch.nn.Parameter(torch.tensor(sampled_points, dtype=torch.float32))
    
    def forward1(self, contour):
        """compute distance in contour pixel space"""

        contour = np.asarray(contour, dtype=np.float32)
        n = len(contour)

        diffs = np.diff(contour, axis=0)
        diffs = np.vstack([diffs, contour[0] - contour[-1]])

        segment_lengths = np.ones(len(diffs), dtype=np.float32)
        cumlen = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_len = cumlen[-1]

        # --- Step 2: choose evenly spaced target distances ---
        target_lens = np.linspace(0, total_len, sh.num_samples, endpoint=False)

        # --- Step 3: interpolate points along contour ---
        sampled_points = []
        for t in target_lens:
            idx = np.searchsorted(cumlen, t, side="right") - 1
            idx = np.clip(idx, 0, n - 1)

            seg_start = contour[idx]
            seg_end = contour[(idx + 1) % n]
            seg_len = 1.0  # since 4-connected step always = 1
            alpha = (t - cumlen[idx]) / seg_len

            pt = (1 - alpha) * seg_start + alpha * seg_end
            
            pt = np.round(pt) + 0.5  # align to pixel centers

            sampled_points.append(pt)

        return np.array(sampled_points, dtype=np.float32)