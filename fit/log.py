from pathlib import Path
import numpy as np
import ipdb
from PIL import Image, ImageDraw

class Logger:
    def __init__(self, flush_every=50):
        self.sublogs = []
        self.iters = []
        self.img_losses = []
        self.smooth_losses = [] 
        self.straight_losses = [] 
        self.axis_align_losses = [] 
        self.curvature_losses = [] 
        self.band_losses = []
        self.losses = []

        self._buffer = []          # hold log lines in memory
        self._flush_every = flush_every
        self.log_path = None

    def create_log(self, log_path):
        self.log_path = log_path
        # make sure file exists (truncate old content)
        with open(self.log_path, 'w'):
            pass

    def print(self, text):
        """Add log text to buffer. Flush periodically."""
        if not text.endswith("\n"):
            text += "\n"
        self._buffer.append(text)

        if len(self._buffer) >= self._flush_every:
            self.flush()

    def flush(self):
        """Write buffer to disk."""
        if self.log_path and self._buffer:
            with open(self.log_path, 'a') as f:
                f.writelines(self._buffer)
            self._buffer.clear()

    def close(self):
        """Flush remaining logs when training ends."""
        self.flush()

    def log_loss(self, iter, img_loss, smooth_loss, band_loss, straight_loss, axis_align_loss, curvature_loss, loss):
        """Save loss values to memory (not flushed)."""
        self.iters.append(iter)
        self.img_losses.append(img_loss)
        self.smooth_losses.append(smooth_loss)
        self.straight_losses.append(straight_loss)
        self.axis_align_losses.append(axis_align_loss)
        self.curvature_losses.append(curvature_loss)
        self.band_losses.append(band_loss)
        self.losses.append(loss)

    def plot_losses(self, save_path, txt_path=None):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.iters, self.img_losses, label='Image Loss')
        plt.plot(self.iters, self.smooth_losses, label='Smoothness Loss')
        plt.plot(self.iters, self.straight_losses, label='Straightness Loss')
        plt.plot(self.iters, self.axis_align_losses, label='Axis-align Loss')
        plt.plot(self.iters, self.curvature_losses, label='Curvature Loss')
        plt.plot (self.iters, self.band_losses, label='Band Loss')
        plt.plot(self.iters, self.losses, label='Total Loss')

        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        # plt.ylim(0, 0.2)
        plt.tight_layout()

        plt.legend()
        plt.savefig(save_path)
        plt.close()

        if txt_path is not None:
            header = "iter img_loss smooth_loss straight_loss axis_align_loss curvature_loss band_loss total_loss"
            data = np.column_stack([
                self.iters,
                self.img_losses,
                self.smooth_losses,
                self.straight_losses,
                self.axis_align_losses,
                self.curvature_losses,
                self.band_losses,
                self.losses
            ])
            np.savetxt(txt_path, data, header=header, comments='')




logger = Logger()


import json
import torch
from pathlib import Path
import math

class GradLogger:
    def __init__(self, summarize=False):
        """
        Args:
            summarize (bool): If True, store gradient statistics (mean, std, norm)
                              instead of full tensors.
        """
        self.logs = []

    def _process_tensor(self, tensor):
        """Convert tensor to list or summary stats."""
        if not isinstance(tensor, torch.Tensor):
            return tensor

        return tensor.detach().cpu()

    def log_iter(self, t, grads_dict):
        """
        Log gradients for a single iteration.

        Args:
            t (int): iteration index
            grads_dict (dict): e.g. {'img_loss_grad': tensor, 'smooth_grad': tensor, ...}
        """
        iter_log = {"iter": t}
        for key, value in grads_dict.items():
            iter_log[key] = self._process_tensor(value)
        self.logs.append(iter_log)

    def save(self, save_path):
        """Save all logged gradients as JSON."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.logs, f, indent=4)
        print(f"✅ Gradient logs saved to {save_path}")


grad_logger = GradLogger()

def visualize_grad(grad_list, save_path="grad_grid.png", 
                        canvas_size=1024, scale=0.3, arrow_color="cyan",
                        bg_color="white", margin=20):
    """
    Visualize gradient vectors arranged in a grid.

    Args:
        grad: [N,2] torch tensor or numpy array of gradients
        save_path: file path to save the image
        canvas_size: width/height of the canvas in pixels
        scale: scales gradient vector length
        arrow_color: color for gradient arrows
        bg_color: background color
        margin: margin around the drawing area
    """



    color_list = ["cyan", "magenta", "yellow", "lime", "orange", "white"]

    # Convert to numpy and ensure list
    grad_list = [g.detach().cpu().numpy() if isinstance(g, torch.Tensor) else np.asarray(g)
                 for g in grad_list]

    # Check all same shape
    N = grad_list[0].shape[0]
    for g in grad_list:
        assert g.shape == (N, 2), f"All gradients must have shape ({N}, 2), got {g.shape}"

    # Grid shape
    n_cols = math.ceil(math.sqrt(N))
    n_rows = math.ceil(N / n_cols)

    grid_w = (canvas_size - 2 * margin) / n_cols
    grid_h = (canvas_size - 2 * margin) / n_rows

    img = Image.new("RGB", (canvas_size, canvas_size), bg_color)
    draw = ImageDraw.Draw(img)

    for i in range(N):
        row = i // n_cols
        col = i % n_cols
        cx = margin + col * grid_w + grid_w / 2
        cy = margin + row * grid_h + grid_h / 2

        # Draw dot at center
        draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill="red")

        # Draw all gradient arrows for this point
        for g, color in zip(grad_list, color_list):
            gx, gy = g[i]
            end_x = cx + gx * scale * canvas_size
            end_y = cy + gy * scale * canvas_size
            draw.line([(cx, cy), (end_x, end_y)], fill=color, width=2)

    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)
    # print(f"Saved multi-gradient visualization → {save_path}")