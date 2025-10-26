from pathlib import Path
from PIL import Image
import diffvg
import pydiffvg
import torch
import skimage
from pathlib import Path
import subprocess
import numpy as np
import torchvision.transforms as T
import cairosvg
from svgpathtools import parse_path
from svgpathtools import Path as svgPath
import shutil
import ipdb
from tqdm import tqdm
import random
import json
import time

from .share import sh
from .utils import points_to_png, points_to_svg, expand_points_param, visualize_angles, points_to_pt, grad_to_pt
from .files import render_fitting_res, visualize_video, create_exp
from .log import logger, Logger, visualize_grad
from .segmentation import sample_from_boundary
from .loss import SmoothnessLoss, BandLoss, ImageLoss, StraightnessLoss
from .trim import add_hard, remove_hard, trim, detect_outlier_edge, insert_closest_band_point, detect_redundant_point_by_edge, remove_redundant_point,\
    detect_redundant_point, remove_redundant_point, remove_points_based_on_loss, add_points_based_on_optimization
from .monitor import Monitor

# object-in-subfolder-level

def train(points_init, points_n, train_path, train_sh):

    sublogger = Logger()

    render = pydiffvg.RenderFunction.apply

    color_n = torch.tensor(sh.color_guess, requires_grad=True)
    polygon = pydiffvg.Polygon(points = points_n, is_closed = True)
    shapes = [polygon]
    polygon_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                        fill_color = color_n)
    shape_groups = [polygon_group]
    polygon.points = points_n * sh.w
    polygon_group.color = color_n

    optimizer = torch.optim.Adam([points_n], lr=1e-3)
    monitor = Monitor()

    for t in range(train_sh.epoch):
        
        with monitor.section("forward rendering"):

            optimizer.zero_grad()
            # Forward pass: render the image.
            shapes[0].points = points_n * sh.w

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
        
        with monitor.section("save_images"):

            # interval = sh.epoch // 10
            interval = 1

            if t >= 0 and t % interval == 0:
            # Save the intermediate render.
                pydiffvg.imwrite(img.cpu(), train_path / 'vis' / 'render_iter_{:03}.png'.format(t), gamma=2.2)
                points_to_png(points_n, train_path / 'vis' / 'iter_{:03}.png'.format(t), \
                              background_image=sh.contour_img, midpoint_indices=None)
                points_to_pt(points_n.cpu(), train_path / 'points' / "points_{:03}.pt".format(t))
                

            if t == 0:
                pydiffvg.imwrite(img.cpu(), train_path / 'init_render.png', gamma=2.2)



        with monitor.section("calculate_loss"):

            img_loss = ImageLoss()(img, sh.raster)
            # img_loss = 2 * (img - raster).pow(2).mean() 
            # smooth_loss = SmoothnessLoss()(points_n, points_init=points_init, is_close=True)
            
            smooth_loss = SmoothnessLoss(train_sh.smooth_loss)(points_n, points_init=points_init, is_close=True)

            band_loss = BandLoss(train_sh.band_loss)(points_n, sh.udf)

            straightness_loss = StraightnessLoss(train_sh.straightness_loss)(points_n, is_close=True)

            img_loss.backward(retain_graph=True)
            points_grad = points_n.grad.clone()
            img_grad = points_n.grad.clone()
            
            # ipdb.set_trace()

            smooth_loss.backward(retain_graph=True)
            smooth_grad = points_n.grad.clone() - points_grad
            # points_grad = points_n.grad.clone()

            band_loss.backward(retain_graph=True)
            band_grad = points_n.grad.clone() - points_grad
            points_grad = points_n.grad.clone()

            straightness_loss.backward()
            straight_grad = points_n.grad.clone() - points_grad

            # ipdb.set_trace()

            loss = img_loss + smooth_loss + band_loss + straightness_loss

            grad_to_pt(img_grad, smooth_grad, band_grad, straight_grad, train_path / 'grad' / "grad_{:03}.pt".format(t))

            # ipdb.set_trace()
            # visualize_angles(points_n.detach().cpu(), is_close=True, save_path=exp_path / "angle.png")
            
            


        with monitor.section("log"):

            if t % 1 == 0:
                logger.print(f'iteration: {t} \n')
                logger.print(f'loss: {loss.item():.6f}, img_loss: {img_loss.item():.6f}, smooth_loss: {smooth_loss.item():.6f} \n')

                sublogger.log_loss(t, img_loss.item(), smooth_loss.item(), band_loss.item(),\
                                straightness_loss.item(), 0, 0,  loss.item())

        
        with monitor.section("backward"):
            # loss.backward()
            optimizer.step()

    visualize_angles(points_n.detach().cpu(), is_close=True, save_path=train_path / "angle.png")
    
    logger.close()
    sublogger.plot_losses(train_path / "loss.png", train_path / "loss.txt")

    monitor.report(train_path / "time_report.json")
    render_fitting_res(shapes, shape_groups, points_n, color_n, save_path=train_path, midpoint_indices=None)

    visualize_video(train_path / "vis", train_path/"vis.mp4", delete_images=False)
    logger.print("-"*40 + "\n\n\n") 

    return points_n

# def trim(points_n):
#     # ipdb.set_trace()
#     redundant_point = detect_redundant_point(points_n)
#     points_n = remove_redundant_point(points_n, redundant_point)

#     redundant_point = detect_redundant_point_by_edge(points_n)
#     points_n = remove_redundant_point(points_n, redundant_point)
    
#     return points_n.detach().clone().requires_grad_(True)

def run(raster_path, exp_path):

    # prepare
    sh.sub_path = exp_path
    to_tensor = T.ToTensor()
    raster = to_tensor(Image.open(raster_path).convert("RGBA")).permute(1, 2, 0)
    # ipdb.set_trace()

    pydiffvg.imwrite(raster.cpu(), exp_path / 'target.png', gamma=2.2)

    sh.w = raster.shape[0]
    logger.print(f"current raster: {raster_path}")

    background = torch.zeros((sh.w, sh.w, 4))
    background[..., 3] = 1.0 
    sh.background = background

    # init shape
    # ipdb.set_trace()
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    points_n, udf, contour_img, contour = sample_from_boundary(exp_path / 'target.png', contour_path=exp_path/'contour.png')
    points_to_png(points_n, exp_path / "init.png", background_image=contour_img)


    sh.contour_img = contour_img; sh.contour = contour; sh.raster = raster
    sh.udf = udf

    points_n, _ = add_hard(points_n, exp_path / "init_after_adding.png")    
    points_n, points_init = remove_hard(points_n, exp_path / "init_vec.png")
    
    points_n = train(points_init, points_n, exp_path / "warmup", sh.warmup_sh)
    # points_loss = remove_points_based_on_loss(points_n, raster)
    # points_to_png(points_n.detach().cpu().numpy(), exp_path / "remove_points_loss_diff.png", point_values=np.array(points_loss))
    ipdb.set_trace()


    # ipdb.set_trace()
    # points_loss = remove_points_based_on_loss(points_n, raster)
    # points_n = add_points_based_on_optimization(points_n, sh.add_points_sh)
    # points_to_png(
    #     points_n.detach().cpu().numpy(), 
    #     exp_path / "trim_points_loss.png", 
    #     point_values=np.array(points_loss)  # This will color points from blue to red
    # )
    # ipdb.set_trace()

    # first pass
    points_n, points_init = trim(points_n)
    points_n = train(points_init, points_n, exp_path / "pass", sh.pass_sh)

    
    # points_to_png(
    #     points_n.detach().cpu().numpy(), 
    #     exp_path / "trim_points_loss.png", 
    #     point_values=np.array(points_loss)  # This will color points from blue to red
    # )
    # ipdb.set_trace()

    # second pass
    points_n, points_init = trim(points_n)
    points_n = train(points_init, points_n, exp_path / "pass_2", sh.pass_sh)

    # points_n = trim(points_n)
    
    # points_n = add_points_based_on_optimization(points_n, sh.add_points_sh)
    points_to_png(points_n, exp_path / "vec_bg.png", background_image=contour_img, midpoint_indices=sh.midpoint_indices)
    points_to_png(points_n, exp_path / "vec.png")




def batch(fold, resolution):
    logger.create_log("./log.txt")
    create_exp()

        # ---- Prepare global metadata ----
    metadata = {
        "id": sh.exp_path.name,
        "created_at": sh.create_at,
        "desription": "",
        "finish": False,
        "resolution": None,
        "is_useless": False,
    }
    
    metadata_path = sh.exp_path / "@metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    subfolds = list(fold.glob("*/"))  # materialize generator so tqdm knows length
    for subfold in tqdm(subfolds, desc="curve fitting"):
        
        if not (subfold / "aa_16.png").exists(): continue

        sh.exp_sub_path = sh.exp_path / subfold.name
        sh.exp_sub_path.mkdir(parents=True, exist_ok=True)

        raster_name = f"aa_{resolution}.png"
        shutil.copy(subfold / raster_name, sh.exp_sub_path)

        raster_path = subfold / f"aa_{resolution}.png"

        run(raster_path, sh.exp_sub_path)

        metadata["resolution"] = sh.w

    metadata["finish"] = True
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)



if __name__ == "__main__":

    sub_path = Path(r"E:\Ziyu\workspace\diff_aa_solution\pipeline\exp\10-22\22-03-14\axe")

    exp_path = sub_path / "test_2"

    sh.sub_path = sub_path

    logger.create_log(sub_path / "log.txt")
    
    if exp_path.exists():
        import shutil
        shutil.rmtree(exp_path)
    exp_path.mkdir(parents=True)

    run(raster_path = sub_path / "aa_64.png",\
        exp_path = exp_path)