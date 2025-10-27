import diffvg
import pydiffvg
import torch
import random

from .share import sh

def primitive(points_n):
    render = pydiffvg.RenderFunction.apply

    color_n = torch.tensor(sh.color_guess, requires_grad=True)
    polygon = pydiffvg.Polygon(points = points_n, is_closed = True)
    shapes = [polygon]
    polygon_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                        fill_color = color_n)
    shape_groups = [polygon_group]
    polygon.points = points_n * sh.w

    polygon_group.color = color_n
    polygon_group.fill_color = color_n


    return render, shapes, shape_groups

def diff_render(render, points_n, shapes, shape_groups, is_random=True):
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        sh.w, sh.w, shapes, shape_groups)
    
    shapes[0].points = points_n * sh.w

    if is_random:
        seed = random.randint(0, 2**31 - 1)
    else: seed = 0

    img = render(sh.w,   # width
                sh.w,   # height
                sh.num_mc_x,     # num_samples_x
                sh.num_mc_y,     # num_samples_y
                seed + 117,   # seed
                sh.background, # background_image
                *scene_args)
    
    return img