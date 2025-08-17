#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Note: The last frame of the episode doesn't always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossy compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Examples:

- Visualize data stored on a local machine:
```
local$ python -m lerobot.scripts.visualize_dataset \
    --repo-id lerobot/pusht \
    --episode-index 0
```

- Visualize data stored on a distant machine with a local viewer:
```
distant$ python -m lerobot.scripts.visualize_dataset \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --save 1 \
    --output-dir path/to/directory

local$ scp distant:path/to/directory/lerobot_pusht_episode_0.rrd .
local$ rerun lerobot_pusht_episode_0.rrd
```

- Visualize data stored on a distant machine through streaming:
(You need to forward the websocket port to the distant machine, with
`ssh -L 9087:localhost:9087 username@remote-host`)
```
distant$ python -m lerobot.scripts.visualize_dataset \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --mode distant \
    --ws-port 9087

local$ rerun ws://localhost:9087
```

"""

import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm
import math
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from scipy.spatial.transform import Rotation as R


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


home_pose = [0,0,0,-2.70767309, -0.67766038, -2.99060844]
TRAJECTORIES_D = [
    # [[0,0,0 ,0,0,0], [0.3,0,0 ,0,0,0], [-0.6,0,0 ,0,0,0], [0.3,0,0 ,0,0,0] ],
    # [[0,0.1,0 ,0,0,0], [0,-0.3,0 ,0,0,0],[0,0.2,0 ,0,0,0] ],
    # [[0,-0.2,0 ,0,0,0], [0,0,0.3 ,0,0,0], [0,0,-0.5 ,0,0,0], [0,0,0.2 ,0,0,0],[0,0.2,0 ,0,0,0], ],

    [[0,0,0 ,0,0,0],[0,0,0 ,np.pi/4,0,0], [0,0,0 ,-np.pi/2,0,0],[0,0,0 ,np.pi/4,0,0], ],
    [[0,0,0 ,0,np.pi/4,0], [0,0,0 ,0,-np.pi/2,0],[0,0,0 ,0,np.pi/4,0]],
    [[0,0,0 ,0,0,np.pi/4], [0,0,0 ,0,0,-np.pi/2],[0,0,0 ,0,0,np.pi/4]],

]


def get_minimum_angle_diff(rot_a, rot_b):
    vect = [[1,0,0],[0,1,0], [0,0,1]]

    vect_a = rot_a.apply(vect)
    print(vect, " -- ", vect_a)
    vect_b = rot_b.apply(vect)
    print(vect, " -- ", vect_b)
    # print("="*100)
    


    rot_diff, rssd = R.align_vectors(vect_a, vect_b)
    assert rssd < 0.01

    return rot_diff.magnitude()

# def inverse_special(r):
#     eu = r.as_euler("xyz")
#     # eu = -1*eu
#     eu[0], eu[1],eu[2] = -eu[0], eu[1], eu[2]
#     return R.from_euler("xyz", eu)

def inverse_special(r):
    rvec = r.as_rotvec()
    # eu = -1*eu
    rvec[0]= -rvec[0]
    return R.from_rotvec(rvec)

def get_rot_offset(robot_frame_pose, umi_pose):
    expected_rot = inverse_special( R.from_euler("xyz", robot_frame_pose[3:]) )

    umi_rot = R.from_euler("xyz", umi_pose[3:])

    rot_off = expected_rot * umi_rot.inv()
    return rot_off.as_euler("xyz")

# def pose_actions_update(move_values, current_pose):
#         # move_values = ds_frame["action.pose"].numpy()
#     # current_pose[:3] += move_values[:3]# /2 # temporary to limit movment
#     # print( move_values[3:])
#     # print( current_pose[3:])
#     current_pose[3:] = inverse_special(
#         R.from_euler("yxz", move_values[3:]) * R.from_euler("yxz", current_pose[3:]) 
#     ).as_euler("yxz")
#     # print(move_values[3:])

#     return current_pose, "pose"

# orentations = []
# def find_min_angle(traj, robot_move_list):
#     pos = home_pose.copy()

#     print(robot_move_list)
#     print(home_pose)
#     robot_ground_truth, _ = pose_actions_update(robot_move_list, home_pose.copy())
#     robot_ground_truth = R.from_euler("xyz", robot_ground_truth[3:])
#     print(robot_ground_truth.as_euler("xyz"))
#     print("+"*100)
#     min_amp = math.inf
#     min_amp_idx = -1

#     for i,row in enumerate(traj):
#         pos,_ = pose_actions_update(row["action.pose"].numpy(), pos)

#         crr_rot = R.from_euler("xyz", pos[3:])
#         amp = get_minimum_angle_diff(robot_ground_truth, crr_rot)

#         if amp < min_amp:
#             min_amp = amp
#             min_amp_idx = i
#     print(min_amp_idx ,"->", min_amp , "  at timestamp:") # traj["timestamp"][min_amp_idx]



def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
    save: bool = False,
    output_dir: Path | None = None,
) -> Path | None:
    if save:
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    repo_id = dataset.repo_id

    logging.info("Loading dataloader")
    episode_sampler = EpisodeSampler(dataset, episode_index)

    print(len(episode_sampler.frame_ids))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    logging.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    spawn_local_viewer = mode == "local" and not save
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)

    # Manually call python garbage collector after `rr.init` to avoid hanging in a blocking flush
    # when iterating on a dataloader with `num_workers` > 0
    # TODO(rcadene): remove `gc.collect` when rerun version 0.16 is out, which includes a fix
    gc.collect()

    if mode == "distant":
        rr.serve(open_browser=False, web_port=web_port, ws_port=ws_port)

    logging.info("Logging to Rerun")

    # print(dataset.meta.features)
    # print(dataset.stats)
    lastPose = [0]*6
    last_robot_rec_pose = home_pose.copy()
    def pose_actions_update(ds_frame, current_pose):

        move_values = ds_frame.numpy()
        current_pose[:3] += move_values[:3]

        move_rot = R.from_euler("xyz", move_values[3:])


        # # -----------------
        # euler_swap = move_rot.as_euler("xyz" )
        # # euler_swap[0],euler_swap[1] = euler_swap[1],euler_swap[0]

        # move_rot2 = R.from_euler("xyz", euler_swap)
        # # -----------------

        crr_inverted_back = inverse_special(R.from_euler("xyz",current_pose[3:]) )
        current_pose[3:] = inverse_special(
            move_rot * crr_inverted_back ).as_euler("xyz")

        return current_pose, "pose"


    def test(ds):
        delta_pos = ds[0]["observation.state.pose"].numpy()
        print("action 0: ", ds[0]["action.pose"])
        nr = 0
        for row in ds:
            delta_pos,_ = pose_actions_update(row["action.pose"], delta_pos)
            if not np.allclose(delta_pos, row["observation.state.pose"].numpy()):
                print("nr = ",nr)
                print(np.isclose(delta_pos, row["observation.state.pose"].numpy()))
                print(delta_pos, " - ",row["observation.state.pose"].numpy())
            assert np.allclose(delta_pos[:3], row["observation.state.pose"].numpy()[:3])

            assert R.from_euler("xyz", delta_pos[3:]).approx_equal( R.from_euler("xyz", row["observation.state.pose"].numpy()[3:]) )
            nr+=1

    # test(dataset)
    # ep1 -> traj 0
    # ep 2 -> traj 1
    # ep 3 -> traj 2
    # find_min_angle(dataset,TRAJECTORIES_D[0][1])
    offsets =[]
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        # print(batch)
        # iterate over the batch
        for i in range(len(batch["index"])):
            # print(batch["task"][i])
            # print(batch["prompt"][i])
            # rr.set_time_sequence("frame_index", batch["frame_index"][i].item())
            # rr.set_time_seconds("timestamp", batch["timestamp"][i].item())
            
            # print("i === ",i)
            # display each camera image
            for key in dataset.meta.camera_keys:
                # TODO(rcadene): add `.compress()`? is it lossless?
                rr.log(key, rr.Image(to_hwc_uint8_numpy(batch[key][i])))

            # display each dimension of action space (e.g. actuators command)
            if "action" in batch:
                for dim_idx, val in enumerate(batch["action"][i]):
                    rr.log(f"action/{dim_idx}", rr.Scalar(val.item()))

            # display each dimension of observed state space (e.g. agent position in joint space)
            if "observation.state.pose" in batch:
                for dim_idx, val in enumerate(batch["observation.state.pose"][i]):
                    name = dataset.meta.features["observation.state.pose"]["names"][dim_idx]

                    
                    # when to show delta/relative pos
                    if False:
                        lastPose[dim_idx] = lastPose[dim_idx] + val.item()
                        print_val = lastPose[dim_idx]
                    else:
                        print_val = val.item()


                    rr.log(f"state/{name}", rr.Scalar(print_val))

            # print(batch["observation.state.gripper"][i])
            if "action.gripper" in batch:
                rr.log(f"state/gripper", rr.Scalar(batch["action.gripper"][i].item()))

            if "action.pose" in batch:
                 lastPose,_ = pose_actions_update(batch["action.pose"][i], lastPose)
                 last_robot_rec_pose ,_ = pose_actions_update(batch["action.pose"][i], last_robot_rec_pose)

                 rot_off = get_rot_offset(last_robot_rec_pose, batch["observation.state.pose"][i])
                 offsets.append(R.from_euler("xyz", rot_off ) )
                 for dim_idx, val in enumerate(batch["action.pose"][i]):
                    name = dataset.meta.features["action.pose"]["names"][dim_idx]


                    print_val = lastPose[dim_idx]

                    print_robot_val = last_robot_rec_pose[dim_idx]

                    rr.log(f"action.pose/{name}", rr.Scalar(print_val))
                    rr.log(f"robot.pose/{name}", rr.Scalar(print_robot_val))
                    if dim_idx > 2:
                        rr.log(f"offset/{name}", rr.Scalar(rot_off[dim_idx-3]))

            if "next.done" in batch:
                rr.log("next.done", rr.Scalar(batch["next.done"][i].item()))

            if "next.reward" in batch:
                rr.log("next.reward", rr.Scalar(batch["next.reward"][i].item()))

            if "next.success" in batch:
                rr.log("next.success", rr.Scalar(batch["next.success"][i].item()))
    
    off_0 = offsets[0]
    ii = 0
    for rot in offsets:
        
        if not rot.approx_equal(off_0):
            print(ii)
        ii+=1


    if mode == "local" and save:
        # save .rrd locally
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        return rrd_path

    elif mode == "distant":
        # stop the process from exiting since it is serving the websocket connection
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="Episode to visualize.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write a .rrd file when `--save 1` is set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes of Dataloader for loading the data.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        help=(
            "Mode of viewing between 'local' or 'distant'. "
            "'local' requires data to be on a local machine. It spawns a viewer to visualize the data locally. "
            "'distant' creates a server on the distant machine where the data is stored. "
            "Visualize the data by connecting to the server with `rerun ws://localhost:PORT` on the local machine."
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Web port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=9087,
        help="Web socket port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help=(
            "Save a .rrd file in the directory provided by `--output-dir`. "
            "It also deactivates the spawning of a viewer. "
            "Visualize the data by running `rerun path/to/file.rrd` on your local machine."
        ),
    )

    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LeRobotDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    args = parser.parse_args()
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")

    logging.info("Loading dataset") #episodes=range(50),
    dataset = LeRobotDataset(repo_id,  root=root, tolerance_s=tolerance_s, force_cache_sync=True)

    visualize_dataset(dataset, **vars(args))


if __name__ == "__main__":
    main()