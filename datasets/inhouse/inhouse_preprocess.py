"""
Convert an in-house (motif/alchera) log into drivestudio's processed layout.

Everything needed is already in one place: the anno-3DOD export carries the
sensor calibration, the per-frame lidar pose and the tracked 3D boxes, while the
images come from `perception_sensor_data_undist` and the point clouds from
`perception_sensor_data/lidar-top/*.laz`.

The undistorted images were NOT rectified with the fisheye K of the calibration:
they were produced with cv2.fisheye.estimateNewCameraMatrixForUndistortRectify
(balance=0), i.e. a new camera matrix with f ~= 960 px instead of ~1190 px
(verified by re-undistorting raw frames: pixel-identical only with balance=0).
That matrix is recomputed here from `*__camera_config.json` (fisheye K + k1..k4)
and written as the intrinsics; using the raw fisheye K gives a ~19% radial
scale error that grows towards the image borders.

Frames follow PandaSet's conventions so that `PandaPixelSource` can be reused
verbatim -- the camera axes are already OpenCV (x right, y down, z forward), and
object class names are remapped onto the PandaSet vocabulary that
`OBJECT_CLASS_NODE_MAPPING` expects.

Coordinate chain
    world      == lidar frame of the first exported frame (lidar_pose is identity there)
    ego        == lidar   (as in the PandaSet processor, the lidar pose is the ego pose)
    c2w[t]     == lidar_pose[t] @ inv(T_ego_lidar) @ T_ego_cam
    box -> world  == lidar_pose[t] @ box_in_lidar

Usage:
    python -m datasets.inhouse.inhouse_preprocess \
        --seq_dir /mnt/ssd2/inhouse/alchera_0723/Vc6fd344-20260629_125501 \
        --target_dir data/inhouse/processed --scene_idx 0 \
        --start_frame 300 --num_frames 150 \
        [--camera_config /path/to/Io52448-S0102-C260602__camera_config.json]
    (camera_config defaults to the *__camera_config.json next to seq_dir's parent)
"""

import argparse
import glob
import json
import os
import re
import shutil

import cv2
import laspy
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# Camera order mirrors PandaSet: front, front-left, front-right, left, right, back.
CAM_ORDER = [
    ("CAM_FRONT", "cam-rgb-front"),
    ("CAM_FRONT_LEFT", "cam-rgb-front_left"),
    ("CAM_FRONT_RIGHT", "cam-rgb-front_right"),
    ("CAM_REAR_LEFT", "cam-rgb-back_left"),
    ("CAM_REAR_RIGHT", "cam-rgb-back_right"),
    ("CAM_REAR", "cam-rgb-back"),
]

# In-house label -> PandaSet vocabulary understood by OBJECT_CLASS_NODE_MAPPING.
# object::* (bollard, traffic_cone, ...) is deliberately absent: it is static
# street furniture and belongs to the background, not to a dynamic node.
CLASS_MAP = {
    "vehicle::car": "Car",
    "vehicle::truck": "Medium-sized Truck",
    "vehicle::bus": "Bus",
    "vehicle::construction_vehicle": "Other Vehicle - Construction Vehicle",
    "vehicle::emergency_vehicle": "Emergency Vehicle",
    "pedestrian::pedestrian": "Pedestrian",
    "twowheeler::bicycle": "Bicycle",
    "twowheeler::motorcycle": "Motorcycle",
    "twowheeler::twowheeler": "Bicycle",
}
NONRIGID = {"Pedestrian", "Bicycle"}
RIGID = {"Car", "Medium-sized Truck", "Bus",
         "Other Vehicle - Construction Vehicle", "Emergency Vehicle", "Motorcycle"}

STATIONARY_THRESH_M = 1.0   # a track that never moves this far is background clutter
Z_NEAR_M = 0.5              # closer than this in front of the camera cannot be projected

# the 12 edges of box_corners(), whose first 4 corners are the +length face and
# whose last 4 are the -length face, each in the same (y, z) winding order
BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7))


def pose_to_matrix(p):
    """{'translation': [x,y,z], 'quaternion': [x,y,z,w]} -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(p["quaternion"], dtype=np.float64)).as_matrix()
    T[:3, 3] = np.asarray(p["translation"], dtype=np.float64)
    return T


def box_to_matrix(b):
    c, s = np.cos(b["yaw"]), np.sin(b["yaw"])
    return np.array([[c, -s, 0, b["x"]],
                     [s,  c, 0, b["y"]],
                     [0,  0, 1, b["z"]],
                     [0,  0, 0, 1]])


def box_corners(b):
    """8 corners of a center-form box, in the box's own frame -> lidar frame."""
    l, w, h = b["length"], b["width"], b["height"]
    x = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * l / 2
    y = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    z = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * h / 2
    pts = np.stack([x, y, z, np.ones(8)])
    return (box_to_matrix(b) @ pts)[:3].T


def clip_box_to_near_plane(pc, z_near=Z_NEAR_M):
    """Clip the 8 camera-frame box corners against the z = z_near plane.

    A box that straddles the image plane - a vehicle right next to the ego car - has
    corners with z <= 0 that cannot be projected. Skipping the whole box (the previous
    behaviour) left exactly the largest, closest vehicles unmasked, which then told the
    training that they were background. Instead keep every corner already in front of
    the plane and, for each edge that crosses it, add the crossing point. The convex
    hull of the result is the correct clipped silhouette.

    Returns None if the box is entirely behind the plane.
    """
    front = pc[:, 2] > z_near
    if not front.any():
        return None
    if front.all():
        return pc

    kept = [pc[front]]
    for i, j in BOX_EDGES:
        if front[i] == front[j]:
            continue
        a, b = pc[i], pc[j]
        # linear interpolation along the edge to the exact z = z_near crossing
        t = (z_near - a[2]) / (b[2] - a[2])
        kept.append((a + t * (b - a))[None])
    return np.concatenate(kept, axis=0)


def frame_index(path):
    m = re.search(r"__f(\d+)__", os.path.basename(path))
    return int(m.group(1)) if m else -1


class InhouseProcessor:
    def __init__(self, seq_dir, save_dir, scene_idx, start_frame, num_frames, cam_ids,
                 camera_config=None):  # CHANGED: fisheye camera config needed for the true undistorted K
        self.seq_dir = seq_dir
        self.scene_dir = os.path.join(save_dir, f"{scene_idx:03d}")
        self.cams = [CAM_ORDER[i] for i in cam_ids]

        ann_files = glob.glob(os.path.join(seq_dir, "annotation", "anno-3DOD-*", "*.json"))
        if not ann_files:
            raise FileNotFoundError(f"no anno-3DOD json under {seq_dir}")
        # prefer the 'visionary' export when several are present
        ann_files.sort(key=lambda p: ("visionary" not in p, p))
        print(f"using annotation: {os.path.basename(ann_files[0])}")
        self.ann = json.load(open(ann_files[0]))
        self.calib = self.ann["calibration"]
        self.frames = self.ann["frames"]

        avail = sorted(int(k) for k in self.frames)
        self.src_frames = [f for f in avail if f >= start_frame][:num_frames]
        if len(self.src_frames) < num_frames:
            print(f"warning: only {len(self.src_frames)} frames available from {start_frame}")
        print(f"frames {self.src_frames[0]}..{self.src_frames[-1]} ({len(self.src_frames)})")

        self.T_ego_lidar = np.array(self.calib["LIDAR_TOP"]["T_parent_child"])
        # ADDED: fisheye intrinsics (K, k1..k4) of the raw cameras; the undistorted images use the
        # balance=0 new camera matrix derived from them, not the fisheye K stored in the annotation
        if camera_config is None:
            cands = glob.glob(os.path.join(os.path.dirname(os.path.normpath(seq_dir)), "*__camera_config.json"))
            if len(cands) != 1:
                raise FileNotFoundError(f"expected one *__camera_config.json next to {seq_dir}, found {cands}; pass --camera_config")
            camera_config = cands[0]
        self.fisheye = {c["Position"]: c for c in json.load(open(camera_config))}
        print(f"using camera config: {os.path.basename(camera_config)}")
        self.lidar_files = {frame_index(p): p for p in
                            glob.glob(os.path.join(seq_dir, "perception_sensor_data",
                                                   "lidar-top", "*.laz"))}
        self.img_files = {}
        for sid, dirname in self.cams:
            self.img_files[sid] = {
                frame_index(p): p for p in
                glob.glob(os.path.join(seq_dir, "perception_sensor_data_undist",
                                       dirname, "*.jpg"))}

    # ------------------------------------------------------------------ setup
    def make_dirs(self):
        for sub in ["images", "lidar", "ego_pose", "extrinsics", "intrinsics",
                    "sky_masks", "instances",
                    "dynamic_masks/all", "dynamic_masks/human", "dynamic_masks/vehicle"]:
            os.makedirs(os.path.join(self.scene_dir, sub), exist_ok=True)

    def cam_matrices(self):
        """Per-camera (K, T_lidar_cam) with T_lidar_cam mapping camera -> lidar."""
        out = []
        inv_ego_lidar = np.linalg.inv(self.T_ego_lidar)
        for sid, _ in self.cams:
            c = self.calib[sid]
            K = self.undistorted_K(sid, c["width"], c["height"])  # CHANGED: was np.array(c["intrinsics"]) (raw fisheye K, wrong for the undistorted images)
            T_lidar_cam = inv_ego_lidar @ np.array(c["T_parent_child"])
            out.append((K, T_lidar_cam, c["width"], c["height"]))
        return out

    def undistorted_K(self, sid, width, height):
        """ADDED: camera matrix of the provided undistorted images = fisheye balance-0 new camera matrix."""
        i = self.fisheye[sid]["intrinsic"]
        K_fish = np.array([[i["fx"], i["skew"], i["cx"]], [0.0, i["fy"], i["cy"]], [0.0, 0.0, 1.0]])
        D = np.array([i["k1"], i["k2"], i["k3"], i["k4"]])
        return cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K_fish, D, (width, height), np.eye(3), balance=0.0)

    # -------------------------------------------------------------- per frame
    def convert(self):
        self.make_dirs()
        cams = self.cam_matrices()

        # intrinsics are constant over the log
        for idx, (K, _, _, _) in enumerate(cams):
            np.savetxt(os.path.join(self.scene_dir, "intrinsics", f"{idx}.txt"),
                       [K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0., 0., 0., 0., 0.])

        instances_info = self.build_instances()

        for t, src in enumerate(tqdm(self.src_frames, desc="frames")):
            l2w = pose_to_matrix(self.frames[str(src)]["lidar_pose"])
            np.savetxt(os.path.join(self.scene_dir, "ego_pose", f"{t:03d}.txt"), l2w)

            for idx, (K, T_lidar_cam, W, H) in enumerate(cams):
                sid = self.cams[idx][0]
                shutil.copyfile(self.img_files[sid][src],
                                os.path.join(self.scene_dir, "images", f"{t:03d}_{idx}.jpg"))
                np.savetxt(os.path.join(self.scene_dir, "extrinsics", f"{t:03d}_{idx}.txt"),
                           l2w @ T_lidar_cam)

            self.save_lidar(src, t)
            self.save_dynamic_masks(src, t, cams, instances_info)

        self.save_instances(instances_info)

    def save_lidar(self, src, t):
        las = laspy.read(self.lidar_files[src])
        pts = np.vstack([las.x, las.y, las.z]).T          # already in the lidar frame
        intensity = np.asarray(las.intensity, dtype=np.float32)
        try:
            laser_id = np.asarray(las.scanner_channel, dtype=np.float32)
        except AttributeError:
            laser_id = np.zeros(len(pts), dtype=np.float32)
        np.column_stack([pts, intensity, laser_id]).astype(np.float32).tofile(
            os.path.join(self.scene_dir, "lidar", f"{t:03d}.bin"))

    def save_dynamic_masks(self, src, t, cams, instances_info):
        objs = [o for o in self.frames[str(src)]["objects"]
                if o["class"] in CLASS_MAP
                and str(o["track_id"]) in instances_info
                and not instances_info[str(o["track_id"])]["_stationary"]]

        for idx, (K, T_lidar_cam, W, H) in enumerate(cams):
            T_cam_lidar = np.linalg.inv(T_lidar_cam)
            masks = {k: np.zeros((H, W), np.uint8) for k in ("all", "human", "vehicle")}
            for o in objs:
                name = CLASS_MAP[o["class"]]
                corners = box_corners(o["box"])
                pc = (np.c_[corners, np.ones(8)] @ T_cam_lidar.T)[:, :3]
                pc = clip_box_to_near_plane(pc)
                if pc is None:                       # entirely behind the image plane
                    continue
                uv = (K @ pc.T).T
                uv = uv[:, :2] / uv[:, 2:3]
                if uv[:, 0].max() < 0 or uv[:, 0].min() > W or \
                   uv[:, 1].max() < 0 or uv[:, 1].min() > H:
                    continue
                # a corner sitting just in front of the near plane projects far outside
                # the frame; clamp to a generous margin so the int32 hull stays sane
                uv = np.clip(uv, [-W, -H], [2 * W, 2 * H])
                hull = cv2.convexHull(uv.astype(np.int32))
                for k in ("all", "human" if name in NONRIGID else "vehicle"):
                    cv2.fillConvexPoly(masks[k], hull, 255)
            for k, m in masks.items():
                Image.fromarray(m).save(
                    os.path.join(self.scene_dir, "dynamic_masks", k, f"{t:03d}_{idx}.png"))

    # --------------------------------------------------------------- objects
    def build_instances(self):
        info = {}
        for t, src in enumerate(self.src_frames):
            l2w = pose_to_matrix(self.frames[str(src)]["lidar_pose"])
            for o in self.frames[str(src)]["objects"]:
                name = CLASS_MAP.get(o["class"])
                if name is None:
                    continue
                sid = str(o["track_id"])
                if sid not in info:
                    info[sid] = dict(id=sid, class_name=name, sibling_id=-1,
                                     frame_annotations={"frame_idx": [], "obj_to_world": [],
                                                        "box_size": [], "stationary": []})
                b = o["box"]
                o2w = l2w @ box_to_matrix(b)
                info[sid]["frame_annotations"]["frame_idx"].append(t)
                info[sid]["frame_annotations"]["obj_to_world"].append(o2w.tolist())
                info[sid]["frame_annotations"]["box_size"].append(
                    [b["length"], b["width"], b["height"]])
                info[sid]["frame_annotations"]["stationary"].append(False)

        # The in-house export has no stationary flag, so derive one: a track whose
        # world-frame centre never travels STATIONARY_THRESH_M is parked scenery.
        n_static = 0
        for v in info.values():
            xyz = np.array([m[:3] for m in
                            [np.array(a)[:3, 3] for a in v["frame_annotations"]["obj_to_world"]]])
            moved = np.linalg.norm(xyz.max(0) - xyz.min(0)) if len(xyz) > 1 else 0.0
            static = moved < STATIONARY_THRESH_M
            v["_stationary"] = bool(static)
            v["frame_annotations"]["stationary"] = [bool(static)] * len(xyz)
            n_static += static
        print(f"INFO: {n_static} static tracks flagged, {len(info) - n_static} moving")
        return info

    def save_instances(self, info):
        keep = {k: {kk: vv for kk, vv in v.items() if kk != "_stationary"}
                for k, v in info.items() if not v["_stationary"]}
        print(f"INFO: Final number of objects: {len(keep)}")

        # The source loader indexes per-instance arrays by int(key), so the keys must
        # form a contiguous 0..N-1 range. Raw track ids leave holes once the static
        # tracks are dropped, and the loader then indexes out of bounds. Renumber,
        # exactly as the PandaSet processor's "correct id" step does.
        id_map = {k: i for i, k in enumerate(keep)}
        renumbered = {id_map[k]: v for k, v in keep.items()}

        frame_instances = {}
        for t in range(len(self.src_frames)):
            frame_instances[t] = [i for k, i in id_map.items()
                                  if t in keep[k]["frame_annotations"]["frame_idx"]]

        d = os.path.join(self.scene_dir, "instances")
        with open(os.path.join(d, "instances_info.json"), "w") as fp:
            json.dump(renumbered, fp, indent=4)
        with open(os.path.join(d, "frame_instances.json"), "w") as fp:
            json.dump(frame_instances, fp, indent=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", required=True)
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--scene_idx", type=int, default=0)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=150)
    ap.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--camera_config", default=None,
                    help="*__camera_config.json (fisheye K, k1..k4); default: the one next to seq_dir's parent")  # ADDED
    args = ap.parse_args()

    InhouseProcessor(args.seq_dir, args.target_dir, args.scene_idx,
                     args.start_frame, args.num_frames, args.cams,
                     camera_config=args.camera_config).convert()  # CHANGED: pass camera config
    print("done ->", os.path.join(args.target_dir, f"{args.scene_idx:03d}"))


if __name__ == "__main__":
    main()
