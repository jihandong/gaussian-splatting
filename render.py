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

import torch
import numpy as np
import json
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
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # Timing statistics for pure render performance
    render_times = []
    total_times = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        render_start = torch.cuda.Event(enable_timing=True)
        render_end = torch.cuda.Event(enable_timing=True)

        iter_start.record()
        render_start.record()
        out = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        render_end.record()

        rendering = out["render"]
        gt = view.original_image[0:3, :, :]

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        # Record timing
        iter_end.record()
        torch.cuda.synchronize()
        render_times.append(render_start.elapsed_time(render_end))  # ms
        total_times.append(iter_start.elapsed_time(iter_end))  # ms

    # Save timing summary
    if render_times:
        timing_summary = {
            "num_frames": len(render_times),
            "render_time_ms": {
                "mean": float(np.mean(render_times)),
                "std": float(np.std(render_times)),
                "min": float(np.min(render_times)),
                "max": float(np.max(render_times)),
                "total": float(np.sum(render_times)),
                "fps": float(1000.0 / np.mean(render_times)) if np.mean(render_times) > 0 else 0.0
            },
            "total_time_ms": {
                "mean": float(np.mean(total_times)),
                "std": float(np.std(total_times)),
                "min": float(np.min(total_times)),
                "max": float(np.max(total_times)),
                "total": float(np.sum(total_times)),
                "fps": float(1000.0 / np.mean(total_times)) if np.mean(total_times) > 0 else 0.0
            },
            "io_overhead_ratio": float((np.mean(total_times) - np.mean(render_times)) / np.mean(total_times)) if np.mean(total_times) > 0 else 0.0
        }
        with open(os.path.join(render_path, "timing_benchmark.json"), "w", encoding="utf-8") as f:
            json.dump(timing_summary, f, ensure_ascii=False, indent=2)
        print(f"\n[Timing] Render: {timing_summary['render_time_ms']['mean']:.2f}ms/frame ({timing_summary['render_time_ms']['fps']:.1f} FPS)")
        print(f"[Timing] Total:  {timing_summary['total_time_ms']['mean']:.2f}ms/frame ({timing_summary['total_time_ms']['fps']:.1f} FPS)")
        print(f"[Timing] I/O overhead: {timing_summary['io_overhead_ratio']*100:.1f}%")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--enable_color_discrimination", action="store_true", help="Enable Color Discrimination early-stop (bit 2)")
    parser.add_argument("--use_mean_T_threshold", action="store_true", help="Use higher early-stop transmittance threshold (~0.015f) via profile bit 3")
    args = get_combined_args(parser)
    # Set profile_mask bits based on convenience flags
    if getattr(args, 'enable_color_discrimination', False):
        try:
            args.profile_mask = int(getattr(args, 'profile_mask', 0)) | 4
        except:
            args.profile_mask = 4
    if getattr(args, 'use_mean_T_threshold', False):
        try:
            args.profile_mask = int(getattr(args, 'profile_mask', 0)) | 8
        except:
            args.profile_mask = 8
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)