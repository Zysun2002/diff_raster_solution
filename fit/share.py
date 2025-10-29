import numpy as np
from pathlib import Path
import torch

class Share:
    def __init__(self):
        pass


sh = Share()



sh.epoch = 200
sh.num_samples = 32

sh.num_mc_x = 4
sh.num_mc_y = 4
sh.color_guess = (1, 1, 1, 1.)
sh.smooth_weight = 10
sh.exp_name = "pipeline"

sh.w = 64

sh.exp_path = Path("./exp/").resolve()


sh.bg_color = (255, 255, 255)         # White (#ffffff)
sh.contour_color = (150, 150, 150)          # Black (#000000)
sh.outline_color = (0, 87, 231)       # Royal blue (#0057e7)
sh.point_color = (215, 38, 61)        # Crimson (#d7263d)
sh.first_point_color = (0, 200, 83)  # Green (#00c853)





sh.warmup_sh = Share()
sh.warmup_sh.epoch = 50

sh.warmup_sh.smooth_loss = {"mode": "external_weight", "w": 0.5}
sh.warmup_sh.band_loss = {"mode": "midpoint", "w": 100}
sh.warmup_sh.straightness_loss = {"mode": None, "w": 0.1}


sh.pass_sh = Share()
sh.pass_sh.epoch = 50

sh.pass_sh.smooth_loss = {"mode": "external_weight", "w": 0.5}
sh.pass_sh.band_loss = {"mode": "midpoint", "w": 100}
sh.pass_sh.straightness_loss = {"mode": "tanh", "w": 1}

sh.add_points_sh = Share()
sh.add_points_sh.epoch = 20

sh.add_points_sh.smooth_loss = {"mode": "angle_weighted", "w": 0.5}
sh.add_points_sh.band_loss = {"mode": "midpoint", "w": 100}
sh.add_points_sh.straightness_loss = {"mode": "tanh", "w": 1}
