"""Export the *time-lapse* of LiDAR point accumulation for individual instances.

`DrivingDataset.get_init_objects` (datasets/driving_dataset.py:263) crops the LiDAR
scan of every frame with the instance's annotated 3D box, transforms the hits into
the object canonical frame and concatenates them across the clip -- but it only ever
hands back the *final* aggregate. This tool replays exactly the same crop while
keeping, for every seed point, the frame it was harvested in, so the aggregation can
be played back frame by frame.

Two accumulation regimes are exported:

  capped     accumulation stops the moment the initialization budget
             (`model.RigidNodes.init.instance_max_pts`, 5000 in configs/omnire.yaml)
             is exhausted -- shows how many frames are enough to fill the budget.
  uncapped   every hit of every frame is kept -- shows what the box crop could have
             delivered if the budget were unlimited.

For reference the actual initialization result is exported too: the real code keeps
*all* frames and then randomly subsamples the whole pool down to 5000, which is a
different set of points than the capped prefix above.

Outputs, per instance, under --out_dir/<dataset>_<scene>/ID=<id>_<class>/
    capped/frame_###.ply     cumulative point cloud, object frame (budget enforced)
    uncapped/frame_###.ply   cumulative point cloud, object frame (no budget)
    init_subsampled.ply      what get_init_objects actually returns
    all_points.ply           every harvested point, colored by birth frame
    stats.json               per-frame counts, poses, box size, moving-filter verdict
and one self-contained `accumulation_viewer.html` per scene with a play button.

Usage:
    python tools/export_instance_accumulation.py \
        --dataset pandaset/6cams --scene_idx 88 --ids 42 44

    python tools/export_instance_accumulation.py \
        --dataset inhouse/6cams --scene_idx 1 --ids 71
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.base.scene_dataset import ModelType
from datasets.driving_dataset import DrivingDataset
from utils.geometry import transform_points
from utils.misc import export_points_to_ply
from tools.accumulation_viewer import write_viewer

NODE_TYPE_NAME = {
    int(ModelType.RigidNodes): "RigidNodes",
    int(ModelType.DeformableNodes): "DeformableNodes",
    int(ModelType.SMPLNodes): "SMPLNodes",
}


def build_cfg(args):
    """Same config assembly as tools/train.py: model config + dataset config + overrides."""
    if args.run_dir is not None:
        cfg = OmegaConf.load(os.path.join(args.run_dir, "config.yaml"))
    else:
        cfg = OmegaConf.load(args.config_file)
        cfg.pop("dataset", None)
        cfg = OmegaConf.merge(
            cfg, OmegaConf.load(os.path.join("configs", "datasets", f"{args.dataset}.yaml"))
        )
    if args.scene_idx is not None:
        cfg.data.scene_idx = args.scene_idx
    if args.start_timestep is not None:
        cfg.data.start_timestep = args.start_timestep
    if args.end_timestep is not None:
        cfg.data.end_timestep = args.end_timestep
    if args.preload_device is not None:
        cfg.data.preload_device = args.preload_device
    return cfg


def harvest(dataset, ins_ids):
    """Replay the per-frame box crop of get_init_objects, keeping frame provenance.

    Returns {ins_id: {"pts": list per frame, "colors": list per frame}} with points in
    the instance's canonical (box-centered, axis-aligned) frame.
    """
    per_frame = {i: {"pts": [], "colors": []} for i in ins_ids}
    pixel = dataset.pixel_source
    for fi in range(dataset.frame_num):
        lidar_dict = dataset.lidar_source.get_lidar_rays(fi)
        lidar_pts = (
            lidar_dict["lidar_origins"]
            + lidar_dict["lidar_viewdirs"] * lidar_dict["lidar_ranges"]
        )
        frame_colors = dataset.lidar_source.colors[lidar_dict["lidar_mask"]]
        for ins_id in ins_ids:
            if not pixel.per_frame_instance_mask[fi, ins_id]:
                per_frame[ins_id]["pts"].append(torch.zeros(0, 3))
                per_frame[ins_id]["colors"].append(torch.zeros(0, 3))
                continue
            # --- identical to datasets/driving_dataset.py:322-337 ---
            o2w = pixel.instances_pose[fi, ins_id]
            o_size = pixel.instances_size[ins_id]
            w2o = torch.inverse(o2w)
            o_pts = transform_points(lidar_pts, w2o)
            mask = (
                (o_pts[:, 0] > -o_size[0] / 2)
                & (o_pts[:, 0] < o_size[0] / 2)
                & (o_pts[:, 1] > -o_size[1] / 2)
                & (o_pts[:, 1] < o_size[1] / 2)
                & (o_pts[:, 2] > -o_size[2] / 2)
                & (o_pts[:, 2] < o_size[2] / 2)
            )
            per_frame[ins_id]["pts"].append(o_pts[mask].cpu())
            per_frame[ins_id]["colors"].append(frame_colors[mask].cpu())
    return per_frame


def traj_length(pixel, ins_id):
    """The moving-object test of get_init_objects (datasets/driving_dataset.py:370-375)."""
    frame_info = pixel.per_frame_instance_mask[:, ins_id]
    trans = pixel.instances_pose[:, ins_id][:, :3, 3][frame_info]
    if trans.shape[0] < 2:
        return 0.0
    return float(torch.norm(trans[1:] - trans[:-1], dim=-1).sum())


def list_all(dataset, max_pts, csv_path=None):
    """How much LiDAR every annotated instance collects, and whether it saturates."""
    pixel = dataset.pixel_source
    ids = list(range(dataset.instance_num))
    per_frame = harvest(dataset, ids)
    info = json.load(open(os.path.join(dataset.data_path, "instances", "instances_info.json")))
    rows = []
    for i in ids:
        counts = np.array([p.shape[0] for p in per_frame[i]["pts"]])
        cum = np.cumsum(counts)
        key = int(pixel.instances_true_id[i])
        sat = int(np.argmax(cum >= max_pts)) + dataset.start_timestep \
            if cum[-1] >= max_pts else None
        rows.append((key, info[str(key)]["class_name"],
                     NODE_TYPE_NAME.get(int(pixel.instances_model_types[i]), "?"),
                     int(pixel.per_frame_instance_mask[:, i].sum()), int(cum[-1]), sat,
                     traj_length(pixel, i)))
    rows.sort(key=lambda r: -r[4])
    print(f"\n{'id':>4}  {'class':<20} {'node':<15} {'frames':>6} {'points':>8} "
          f"{'budget full at':>14}  {'traj m':>7}  moving")
    for key, cls, node, nf, tot, sat, tl in rows:
        print(f"{key:>4}  {cls:<20} {node:<15} {nf:>6} {tot:>8} "
              f"{str(sat) if sat is not None else '-':>14}  {tl:>7.2f}  {'yes' if tl > 1.0 else 'no'}")
    n_sat = sum(1 for r in rows if r[5] is not None)
    print(f"\n{n_sat}/{len(rows)} instances reach the {max_pts}-point budget\n")
    if csv_path:
        with open(csv_path, "w") as f:
            f.write("id,class,node,frames,points,budget_full_at,traj_m\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},"
                        f"{'' if r[5] is None else r[5]},{r[6]:.3f}\n")
        print(f"table written to {csv_path}")


README = """# LiDAR accumulation export -- {scene}

Frame-by-frame replay of the box crop in `DrivingDataset.get_init_objects`
(`datasets/driving_dataset.py:322-342`): at each frame the LiDAR scan is pushed into
the instance's canonical box frame and the hits inside the box are kept, so points
from every frame pile up in one object-local cloud.

`accumulation_viewer.html` -- open in a browser. Play/scrub the accumulation, switch
between the two regimes, colour by LiDAR RGB or by the frame a point arrived in, and
flip between the object frame and the world frame (where the same points show the
trail `filter_pts_in_boxes` carves out of the background seed). Deep link with
`#i=<instance index>&f=<frame>&m=capped|uncapped&c=rgb|age&w=0|1`.

Per instance, in `ID=<id>_<class>/`:

| file | what it is |
|---|---|
| `capped/frame_###.ply` | cumulative cloud, accumulation stops at the {max_pts}-point budget |
| `uncapped/frame_###.ply` | cumulative cloud, no budget |
| `all_points.ply` | every harvested point, LiDAR colour |
| `all_points_by_frame.ply` | same points, tinted by the frame they arrived in |
| `init_subsampled.ply` | what `get_init_objects` actually returns: a random {max_pts}-point draw from the *whole* pool, not the first {max_pts} |
| `stats.json` | per-frame counts, budget saturation frame, trajectory length |

`instance_harvest.csv` (written by `--list`) is the same accounting for every
instance in the scene.

Regenerate with:

    python tools/export_instance_accumulation.py \\
        --config_file {config} --dataset {dsname} --scene_idx {scene_idx} --ids {ids}
"""


def write_readme(scene_dir, payload):
    with open(os.path.join(scene_dir, "README.md"), "w") as f:
        f.write(README.format(
            scene=payload["scene"],
            max_pts=payload["max_pts"],
            config=payload["config"],
            dsname=payload["dsname"],
            scene_idx=payload["scene_idx"],
            ids=" ".join(str(i["id_in_dataset"]) for i in payload["instances"]),
        ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_file", default="configs/omnire.yaml")
    ap.add_argument("--dataset", help="e.g. pandaset/6cams, inhouse/6cams")
    ap.add_argument("--run_dir", help="take the data config from an existing run instead")
    ap.add_argument("--scene_idx")
    ap.add_argument("--start_timestep", type=int)
    ap.add_argument("--end_timestep", type=int)
    ap.add_argument("--preload_device", choices=["cpu", "cuda"],
                    help="override data.preload_device; cpu keeps the images off the GPU")
    ap.add_argument("--ids", nargs="+",
                    help="instance ids as keyed in instances_info.json")
    ap.add_argument("--list", action="store_true",
                    help="harvest every instance and print how much LiDAR each one "
                         "collects (which ones actually saturate the budget); no export")
    ap.add_argument("--max_pts", type=int, default=5000,
                    help="initialization budget, i.e. init.instance_max_pts")
    ap.add_argument("--out_dir", default="debug/instance_accumulation")
    ap.add_argument("--no_frame_plys", action="store_true",
                    help="only write the summary plys, skip the per-frame sequence")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.ids and not args.list:
        raise SystemExit("pass --ids <id ...> or --list")

    torch.manual_seed(args.seed)
    cfg = build_cfg(args)
    dataset = DrivingDataset(data_cfg=cfg.data)
    pixel = dataset.pixel_source

    # instances_true_id is the key of instances_info.json; the index into the dense
    # tables is the *compact* id left after loader dropped out-of-window instances
    true_ids = pixel.instances_true_id.tolist()

    scene_tag = f"{cfg.data.dataset}_{int(cfg.data.scene_idx):03d}" \
        if str(cfg.data.scene_idx).isdigit() else f"{cfg.data.dataset}_{cfg.data.scene_idx}"
    scene_dir = os.path.join(args.out_dir, scene_tag)
    os.makedirs(scene_dir, exist_ok=True)

    if args.list:
        list_all(dataset, args.max_pts, os.path.join(scene_dir, "instance_harvest.csv"))
        if not args.ids:
            return

    targets = {}
    for key in args.ids:
        key = int(key)
        if key not in true_ids:
            raise SystemExit(f"instance {key} is not present in this temporal window")
        targets[int(true_ids.index(key))] = key

    info_path = os.path.join(dataset.data_path, "instances", "instances_info.json")
    instances_info = json.load(open(info_path))

    per_frame = harvest(dataset, list(targets.keys()))
    payload = {
        "scene": scene_tag,
        "start_timestep": int(dataset.start_timestep),
        "num_frames": int(dataset.frame_num),
        "max_pts": int(args.max_pts),
        "config": args.config_file if args.run_dir is None else args.run_dir + "/config.yaml",
        "dsname": args.dataset or "<from run_dir>",
        "scene_idx": cfg.data.scene_idx,
        "instances": [],
    }

    for ins_id, key in targets.items():
        meta = instances_info[str(key)]
        cls = meta["class_name"].replace(" ", "-")
        node_type = NODE_TYPE_NAME.get(int(pixel.instances_model_types[ins_id]), "?")
        o_size = pixel.instances_size[ins_id].cpu().numpy()
        poses = pixel.instances_pose[:, ins_id].cpu().numpy()          # (F, 4, 4)
        visible = pixel.per_frame_instance_mask[:, ins_id].cpu().numpy()

        pts = torch.cat(per_frame[ins_id]["pts"], dim=0).numpy().astype(np.float32)
        cols = torch.cat(per_frame[ins_id]["colors"], dim=0).numpy().astype(np.float32)
        counts = np.array([p.shape[0] for p in per_frame[ins_id]["pts"]], dtype=np.int64)
        cum = np.cumsum(counts)                                        # points after frame fi
        birth = np.repeat(np.arange(dataset.frame_num, dtype=np.int32), counts)

        capped_cum = np.minimum(cum, args.max_pts)
        saturated = int(np.argmax(cum >= args.max_pts)) if cum[-1] >= args.max_pts else None
        tlen = traj_length(pixel, ins_id)

        ins_dir = os.path.join(scene_dir, f"ID={key}_{cls}")
        os.makedirs(ins_dir, exist_ok=True)

        # every harvested point, tinted by the frame it was harvested in
        norm_birth = birth / max(dataset.frame_num - 1, 1)
        heat = np.stack([norm_birth, 0.35 + 0.4 * (1 - norm_birth), 1 - norm_birth], axis=-1)
        export_points_to_ply(torch.from_numpy(pts), torch.from_numpy(cols),
                             os.path.join(ins_dir, "all_points.ply"))
        export_points_to_ply(torch.from_numpy(pts), torch.from_numpy(heat.astype(np.float32)),
                             os.path.join(ins_dir, "all_points_by_frame.ply"))

        # what get_init_objects actually returns: subsample of the FULL pool, not a prefix
        if pts.shape[0] > args.max_pts:
            sel = torch.randperm(pts.shape[0])[: args.max_pts].numpy()
        else:
            sel = np.arange(pts.shape[0])
        export_points_to_ply(torch.from_numpy(pts[sel]), torch.from_numpy(cols[sel]),
                             os.path.join(ins_dir, "init_subsampled.ply"))

        if not args.no_frame_plys:
            for mode, series in (("capped", capped_cum), ("uncapped", cum)):
                mode_dir = os.path.join(ins_dir, mode)
                os.makedirs(mode_dir, exist_ok=True)
                for fi in range(dataset.frame_num):
                    n = int(series[fi])
                    if n == 0:
                        continue
                    export_points_to_ply(
                        torch.from_numpy(pts[:n]), torch.from_numpy(cols[:n]),
                        os.path.join(mode_dir, f"frame_{fi + dataset.start_timestep:03d}.ply"),
                    )

        stats = {
            "id_in_dataset": key,
            "id_in_tables": ins_id,
            "uuid": meta["id"],
            "class_name": meta["class_name"],
            "node_type": node_type,
            "box_size": o_size.tolist(),
            "num_visible_frames": int(visible.sum()),
            "total_points": int(pts.shape[0]),
            "points_per_frame": counts.tolist(),
            "cumulative": cum.tolist(),
            "cumulative_capped": capped_cum.tolist(),
            "budget_saturated_at_local_frame": saturated,
            "budget_saturated_at_frame": None if saturated is None
            else saturated + int(dataset.start_timestep),
            "trajectory_length_m": tlen,
            "passes_only_moving_at_1.0m": bool(tlen > 1.0),
        }
        json.dump(stats, open(os.path.join(ins_dir, "stats.json"), "w"), indent=2)

        print(f"[ID={key}] {meta['class_name']:<8} {node_type:<14} "
              f"visible {int(visible.sum()):>3} frames  total {pts.shape[0]:>6} pts  "
              f"budget hit at {stats['budget_saturated_at_frame']}  traj {tlen:.2f} m")

        payload["instances"].append({
            **stats,
            "pts": pts,
            "colors": (np.clip(cols, 0, 1) * 255).astype(np.uint8),
            "birth": birth,
            "poses": poses.astype(np.float32),
            "visible": visible.astype(np.uint8),
        })

    viewer = os.path.join(scene_dir, "accumulation_viewer.html")
    write_viewer(payload, viewer)
    write_readme(scene_dir, payload)
    print(f"\nwrote {scene_dir}\n  open {viewer} in a browser for the animated view")


if __name__ == "__main__":
    main()
