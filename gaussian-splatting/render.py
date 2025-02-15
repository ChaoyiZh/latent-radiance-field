#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from train import decode_img

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, scene, full_render, use_refine):
    # render latent
    render_latent_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_latent")
    # gt image
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_images")
    # latent_reconstruction
    latent_reconstruction_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_latent_reconstruction")
    # gt_reconstruction
    gts_reconstruction_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_latent_reconstruction")
    # gt_latent
    gts_latent_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_latent")

    makedirs(render_latent_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(latent_reconstruction_path, exist_ok=True)
    makedirs(gts_reconstruction_path, exist_ok=True)
    makedirs(gts_latent_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering_latent = torch.clamp(render(view, gaussians, pipeline, background)["render"], min=0, max=1)
        torchvision.utils.save_image(rendering_latent[0:3, :, :], os.path.join(render_latent_path, '{0:05d}'.format(idx) + ".png"))

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        latent, latent_reconstruction = decode_img(rendering_latent, scene.auto_encoder, scene.latent_scales)
        latent = latent.detach().cpu().numpy()

        np.save(os.path.join(render_latent_path, '{0:05d}'.format(idx) + ".npy"), latent)



        if full_render:
            latent_reconstruction = torch.clamp(latent_reconstruction, 0.0, 1.0)
            torchvision.utils.save_image(latent_reconstruction, os.path.join(latent_reconstruction_path, '{0:05d}'.format(idx) + ".png"))

            gts_reconstruction = torch.clamp(decode_img(view.original_latent_image, scene.auto_encoder, scene.latent_scales)[-1], 0.0,
                                                1.0)
            torchvision.utils.save_image(gts_reconstruction,
                                         os.path.join(gts_reconstruction_path, '{0:05d}'.format(idx) + ".png"))

            gt_latent = view.original_latent_image[0:3, :, :]
            torchvision.utils.save_image(gt_latent,
                                         os.path.join(gts_latent_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, full_render:bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, scene, full_render, dataset.use_refine)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, scene, full_render, dataset.use_refine)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--full_render", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.full_render)