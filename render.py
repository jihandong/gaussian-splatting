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

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        out = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = out["render"]
        gt = view.original_image[0:3, :, :]

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        # Save count profiling heatmaps if present
        if "profile_tests" in out and "profile_contribs" in out:
            tests = out["profile_tests"].cpu()
            contribs = out["profile_contribs"].cpu()
            if train_test_exp:
                w = tests.shape[1]
                tests = tests[:, w // 2:]
                contribs = contribs[:, w // 2:]

            tests_np = tests.numpy()
            contribs_np = contribs.numpy()
            # Raw arrays
            np.save(os.path.join(render_path, f"{idx:05d}_tests.npy"), tests_np)
            np.save(os.path.join(render_path, f"{idx:05d}_contribs.npy"), contribs_np)

            def save_heatmap(arr_np, out_name):
                vmax = max(np.percentile(arr_np, 99.5), 1.0)
                img = np.clip(arr_np.astype(np.float32) / float(vmax), 0.0, 1.0)
                img_t = torch.from_numpy(img).unsqueeze(0)
                torchvision.utils.save_image(img_t, out_name)

            save_heatmap(tests_np, os.path.join(render_path, f"{idx:05d}_tests.png"))
            save_heatmap(contribs_np, os.path.join(render_path, f"{idx:05d}_contribs.png"))

            # Extra analysis: count distribution histograms (uniform bins)
            # We follow the same clipping vmax as heatmap to avoid extreme outliers
            def save_hist(arr_np, out_name, num_bins=10):
                vmax = max(float(np.percentile(arr_np, 99.5)), 1.0)
                edges = np.linspace(0.0, vmax, num_bins + 1, dtype=np.float32)
                hist, _ = np.histogram(arr_np, bins=edges)
                payload = {
                    "num_pixels": int(arr_np.size),
                    "num_bins": int(num_bins),
                    "edges": edges.tolist(),
                    "counts": hist.astype(np.int64).tolist(),
                }
                with open(out_name, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

            save_hist(tests_np, os.path.join(render_path, f"{idx:05d}_tests_hist.json"))
            save_hist(contribs_np, os.path.join(render_path, f"{idx:05d}_contribs_hist.json"))

        # Save timing profiling if present
        if "profile_loop_cycles" in out and "profile_discrim_cycles" in out:
            loop_cycles = out["profile_loop_cycles"].cpu()
            discrim_cycles = out["profile_discrim_cycles"].cpu()
            # Guard empty tensors
            if loop_cycles.numel() > 0 and discrim_cycles.numel() > 0:
                if train_test_exp:
                    w = loop_cycles.shape[1]
                    loop_cycles = loop_cycles[:, w // 2:]
                    discrim_cycles = discrim_cycles[:, w // 2:]

                lc_np = loop_cycles.numpy().astype(np.int64)
                dc_np = discrim_cycles.numpy().astype(np.int64)
                np.save(os.path.join(render_path, f"{idx:05d}_loop_cycles.npy"), lc_np)
                np.save(os.path.join(render_path, f"{idx:05d}_discrim_cycles.npy"), dc_np)

                def save_cycles_heat(arr_np, out_name):
                    if arr_np.size == 0:
                        return
                    vmax = max(float(np.percentile(arr_np, 99.5)), 1.0)
                    img = np.clip(arr_np.astype(np.float32) / float(vmax), 0.0, 1.0)
                    img_t = torch.from_numpy(img).unsqueeze(0)
                    torchvision.utils.save_image(img_t, out_name)

                save_cycles_heat(lc_np, os.path.join(render_path, f"{idx:05d}_loop_cycles.png"))
                save_cycles_heat(dc_np, os.path.join(render_path, f"{idx:05d}_discrim_cycles.png"))

                # Analysis 1: share of total loop time inside discrimination (dc / lc)
                with np.errstate(divide='ignore', invalid='ignore'):
                    share = np.where(lc_np > 0, dc_np / lc_np, 0.0).astype(np.float32)
                if share.size > 0:
                    vmax_share = max(float(np.percentile(share, 99.5)), 1e-6)
                    edges_share = np.linspace(0.0, vmax_share, 11, dtype=np.float32)
                    hist_share, _ = np.histogram(share, bins=edges_share)
                    with open(os.path.join(render_path, f"{idx:05d}_discrim_share_hist.json"), "w", encoding="utf-8") as f:
                        json.dump({
                            "num_pixels": int(share.size),
                            "edges": edges_share.tolist(),
                            "counts": hist_share.astype(np.int64).tolist(),
                            "metric": "dc / lc (fraction of total time)"
                        }, f, ensure_ascii=False, indent=2)

                # Analysis 2: increase ratio relative to baseline without discrimination
                # baseline = lc - dc; increase_ratio = dc / baseline
                with np.errstate(divide='ignore', invalid='ignore'):
                    baseline = (lc_np - dc_np).astype(np.float64)
                    valid = baseline > 0
                    increase_ratio = np.zeros_like(baseline, dtype=np.float32)
                    increase_ratio[valid] = (dc_np[valid].astype(np.float64) / baseline[valid]).astype(np.float32)

                np.save(os.path.join(render_path, f"{idx:05d}_discrim_increase_ratio.npy"), increase_ratio)
                # Heatmap for increase ratio
                if increase_ratio.size > 0:
                    vmax_ir = max(float(np.percentile(increase_ratio[valid] if valid.any() else increase_ratio, 99.5)), 1e-6)
                    img_ir = np.clip((increase_ratio / float(vmax_ir)).astype(np.float32), 0.0, 1.0)
                    torchvision.utils.save_image(torch.from_numpy(img_ir).unsqueeze(0), os.path.join(render_path, f"{idx:05d}_discrim_increase_ratio.png"))

                # Histogram and summary for increase ratio
                if valid.any():
                    data = increase_ratio[valid]
                    vmax_data = max(float(np.percentile(data, 99.5)), 1e-6)
                    edges_ir = np.linspace(0.0, vmax_data, 21, dtype=np.float32)
                    hist_ir, _ = np.histogram(data, bins=edges_ir)
                    # thresholds for counts
                    thresholds = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
                    thr_counts = {str(t): int((data >= t).sum()) for t in thresholds}
                    pct = np.percentile(data, [50, 90, 95, 99, 99.5]).astype(np.float32).tolist()
                    summary = {
                        "total_pixels": int(lc_np.size),
                        "valid_pixels": int(valid.sum()),
                        "mean_loop_cycles": float(lc_np.mean()) if lc_np.size else 0.0,
                        "mean_discrim_cycles": float(dc_np.mean()) if dc_np.size else 0.0,
                        "mean_share": float(share.mean()) if share.size else 0.0,
                        "mean_increase_ratio": float(data.mean()) if data.size else 0.0,
                        "increase_ratio_percentiles": {
                            "p50": pct[0], "p90": pct[1], "p95": pct[2], "p99": pct[3], "p99_5": pct[4]
                        },
                        "increase_ratio_hist": {
                            "edges": edges_ir.tolist(),
                            "counts": hist_ir.astype(np.int64).tolist()
                        },
                        "increase_ratio_threshold_counts": thr_counts
                    }
                    with open(os.path.join(render_path, f"{idx:05d}_timing_summary.json"), "w", encoding="utf-8") as f:
                        json.dump(summary, f, ensure_ascii=False, indent=2)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

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
    parser.add_argument("--enable_pixel_profile", action="store_true", help="Enable per-pixel Gaussian count profiling (bit 0)")
    parser.add_argument("--enable_timing_profile", action="store_true", help="Enable per-pixel timing profiling (bit 1)")
    args = get_combined_args(parser)
    # If user requested the convenience flag, set bit 0 in the profile_mask
    if getattr(args, 'enable_pixel_profile', False):
        try:
            args.profile_mask = int(getattr(args, 'profile_mask', 0)) | 1
        except:
            args.profile_mask = 1
    if getattr(args, 'enable_timing_profile', False):
        try:
            args.profile_mask = int(getattr(args, 'profile_mask', 0)) | 2
        except:
            args.profile_mask = 2
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)