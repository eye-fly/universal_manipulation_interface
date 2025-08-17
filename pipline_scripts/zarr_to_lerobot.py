import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import click
import zarr
from pathlib import Path
import torch
import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub._umi_imagecodecs_numcodecs import register_codecs
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    calculate_episode_data_index,
    concatenate_episodes,
    get_default_encoding,
    save_images_concurrently,
)
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import check_repo_id
from src.lbot.umi_zarr_format import from_raw_to_lerobot_format, umi_feats
from src.lbot.util import offset_rot

def handedness_cor_system(r: R):
    M = np.diag([1, -1, 1])  # Reflect Y / Reverse Roll
    mr = M @ r.as_matrix() @ M
    return R.from_matrix(mr)

fps = 10

import numcodecs
from imagecodecs.numcodecs import Jpegxl
numcodecs.register_codec(Jpegxl)

def load_zarr(zarr_path, dataset):
    zarr_data = zarr.open(zarr_path, mode="r")

    eff_pos = torch.from_numpy(zarr_data["data/robot0_eef_pos"][:])
    eff_rot_axis_angle = torch.from_numpy(zarr_data["data/robot0_eef_rot_axis_angle"][:])
    states_pos = torch.cat([eff_pos, eff_rot_axis_angle], dim=1).numpy()

    gripper_width = (zarr_data["data/robot0_gripper_width"][:])

    episode_ends = zarr_data["meta/episode_ends"][:]
    num_episodes = episode_ends.shape[0]
    # We convert it in torch tensor later because the jit function does not support torch tensors
    episode_ends = torch.from_numpy(episode_ends)

    from_ids, to_ids = [], []
    from_idx = 0
    for to_idx in episode_ends:
        from_ids.append(from_idx)
        to_ids.append(to_idx)
        from_idx = to_idx

    for  selected_ep_idx in tqdm.tqdm(range(num_episodes)):
        from_idx = from_ids[selected_ep_idx]
        to_idx = to_ids[selected_ep_idx]

        last_pose = None
        for frame_idx in range(from_idx,to_idx):
            frame = dict()
            frame["task"] = "put cup on a saucer"

            crr_pose = states_pos[frame_idx]
            

            rot = R.from_rotvec(crr_pose[3:])

            # change_of_basis =  np.matrix('0 1 0; 1 0 0 ; 0 0 1')
            change_of_basis = R.from_euler('xyz',[0,0,np.pi/2 ]).as_matrix()
            rot_matrix = inv(change_of_basis) @ rot.as_matrix() @ change_of_basis
            rot = R.from_matrix(rot_matrix)

            # rot = handedness_cor_system(rot)

            crr_pose[3:] = rot.as_euler('xyz')
            crr_pose[0] = -crr_pose[0]
            crr_pose[1] = -crr_pose[1]

            frame["observation.umi.state.pose"] = crr_pose
            frame["observation.state.pose"]  = offset_rot(crr_pose)

            frame["action.gripper"] = gripper_width[frame_idx]
            frame["observation.state.gripper"]= gripper_width[frame_idx]
            
            frame["action.pose"] = np.zeros_like(crr_pose)
            if not (last_pose is None):
                frame["action.pose"][:3] = crr_pose[:3] - last_pose[:3]

                delta_rot = R.from_euler( "xyz", crr_pose[3:]) * R.from_euler( "xyz", last_pose[3:]).inv()
                frame["action.pose"][3:] = delta_rot.as_euler("xyz")



        

            frame["observation.images"] = zarr_data["data/camera0_rgb"][frame_idx]
            tpf = 1.0/fps
            frame["timestamps"] = np.array([tpf*frame_idx]).astype('float32')

            dataset.add_frame(frame)
            last_pose = crr_pose
        dataset.save_episode()
        # break
    dataset.push_to_hub(private=True)


@click.command()
@click.argument('input', nargs=-1)
@click.option('-rp', '--repo_id', required=True, help='hf dataset name')
@click.option('-p', '--zarr_path', required=True, help='hf dataset name')
def main(input, repo_id, zarr_path):
    use_video = True #TODO
    out_res = (224,224)

    check_repo_id(repo_id)
    


    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps = fps,
        features=umi_feats(*out_res),
        use_videos=use_video,
    )

    load_zarr(zarr_path,dataset )

if __name__ == "__main__":
    main()

