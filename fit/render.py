import diffvg
import pydiffvg
import torch

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