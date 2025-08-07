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
import math 
from numpy.linalg import inv

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def plot_rotated_axes(ax, r, name=None, offset=(0, 0, 0), scale=1):

    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB

    loc = np.array([offset, offset])

    for i, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),

                                      colors)):

        axlabel = axis.axis_name

        axis.set_label_text(axlabel)

        axis.label.set_color(c)

        axis.line.set_color(c)

        axis.set_tick_params(colors=c)

        line = np.zeros((2, 3))

        line[1, i] = scale

        line_rot = r.apply(line)

        line_plot = line_rot + loc

        ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c)

        text_loc = line[1]*1.2

        text_loc_rot = r.apply(text_loc)

        text_plot = text_loc_rot + loc[0]

        ax.text(*text_plot, axlabel.upper(), color=c,

                va="center", ha="center")

    ax.text(*offset, name, color="k", va="center", ha="center",

            bbox={"fc": "w", "alpha": 0.8, "boxstyle": "circle"})


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

def change_cor_system(r):
    rvec = r.as_rotvec()
    rvec[0], rvec[1],rvec[2] = rvec[0], -rvec[1], rvec[2]
    return R.from_rotvec(rvec)
    # eu = r.as_euler("xyz")
    # eu[0], eu[1],eu[2] = eu[0], eu[1], eu[2]
    # return R.from_euler("xyz", eu)

def inverse_special(r):
    eu = r.as_euler("xyz")
    # eu = -1*eu
    eu[0], eu[1],eu[2] = eu[0], -eu[1], eu[2]
    return R.from_euler("xyz", eu)


def handedness_cor_system(r: R):
    M = np.diag([1, -1, -1])  # Reflect X / Reverse Pich
    mr = M @ r.as_matrix() @ inv(M)
    return R.from_matrix(mr)

def reversedMatrixZ(ang):
    return np.matrix([[1, 0, 0], [0, math.cos(ang), math.sin(ang)], [0, -math.sin(ang), math.cos(ang)]] )  

def handedness_cor_system(r: R):
    eu = r.as_euler("xyz")

    X = R.from_euler('x',eu[0]).as_matrix() #reversedMatrixX(-eu[0])
    Y = R.from_euler('y',eu[1]).as_matrix()
    Z = R.from_euler('z',eu[2]).as_matrix()
    Z = reversedMatrixZ(eu[2]) # r.from_euler('z',eu[2]).as_matrix()

    # M = np.diag([-1, 1, 1])  # Reflect X / Reverse Pich
    mr = Z @ Y@ X
    return R.from_matrix(mr)
    
def rvec_filp(r):
    rvec = r.as_rotvec()
    rvec[0], rvec[1],rvec[2] = -rvec[0], -rvec[1], -rvec[2]
    return R.from_rotvec(rvec)

# def handedness_cor_system(r: R):
#     M =  np.matrix('0 1 0; 1 0 0 ; 0 0 1')


#     mr = (M) @ r.as_matrix() @ inv(M)
#     # return R.from_matrix(mr)
#     eu = R.from_matrix(mr).as_euler("xyz")
#     # eu = -1*eu
#     eu[0], eu[1],eu[2] = -eu[0], -eu[1], -eu[2]
#     return R.from_euler("xyz", eu)

# i need to change direction of 1 rotation axis or swap 2 rotation axis with each other

# multiple simmilar question 
# https://stackoverflow.com/questions/1263072/changing-a-matrix-from-right-handed-to-left-handed-coordinate-system/39519079#39519079
# https://stackoverflow.com/questions/1274936/flipping-a-quaternion-from-right-to-left-handed-coordinates
# but bouth 


# np.diag([1, 1, -1])  # Reflect X/Z plane
# this is "equvalent" to rotation around z about angle pi, even 
# thou determinant of matrix is different bouth of those operation 
# resoult is swapped direction of 2 axis of rotation (in this case x->-x, y->-y)
# handedness_cor_system(rot1_crr).approx_equal(rot1_crr) is true for "all" 
def handedness_cor_system(r: R):
    M =  np.matrix('1 0 0; 0 1 0 ; 0 0 -1')
    rot = R.from_euler("z",np.pi).as_matrix()
    M = rot @ M

    mr = (M) @ r.as_matrix() @ inv(M)
    return R.from_matrix(mr)

# heddenes of the cord system during conversion to euler ang changes sing for all euler angles
# https://github.com/benvanik/oculus-sdk/blob/master/LibOVR/Src/Kernel/OVR_Math.h function GetEulerAngles
# but doing this and combining multiple rotations resoults in errors, 
# my best guess is after converting back to R object assumption that euler_angles are in right-handed system is broken and resoults in incorrect representation
def inverse_euler(r):
    eu = r.as_euler("xyz")
    eu *= -1
    return R.from_euler("xyz", eu)


# so currently the system isn't even left handed (in therm of rotation) because on left-handed system (+) rotation is clockwise
# [opposite when in right-handed]
# so no matter if in right-handed or left-handed (+pi/2) from [1,0,0] -> [0,1,0] while in current robot setpu it will give [0,-1, 0]

def handedness_cor_system(r: R):
    quat = r.as_quat()
    # return R.from_quat([quat[1], -quat[2], -quat[0],quat[3]] )
    return R.from_quat([quat[0], -quat[1], quat[2], -quat[3]] )



rot1_last = None
rot1_crr = R.from_euler('xyz',[0, 0,0 ])

rot2_last = None
rot2_crr = R.from_euler('xyz',[0, 0,0 ])
rot2_sim = None
def pnt(initial, i,j, debug = False):
    names = ["roll", "pich",  "yaw"]
    rot_pich = R.from_euler('xyz',[0, (np.pi/2)*(1.0*(i%50)/100),0 ])
    rot_pich = R.from_euler('xyz',[(np.pi/2)*(1.0*i/100),0,0 ])
    rot_pich = R.from_euler('xyz',[0, (np.pi/2)*(1.0*i/100),0 ])
    # rot_pich = R.from_euler('xyz',[0, 0,(np.pi/2)*(1.0*i/100) ])
    rot_roll = R.from_euler('xyz',[ (np.pi/2)*(1.0*j/100),(np.pi/2)*(0.75*j/100) , (np.pi/2)*(0.5*j/100) ])

    rot =  rot_roll *rot_pich* initial
# =============================================

    # assert pos_rot_pich.approx_equal(rot_pich.inv())

    global rot1_last,rot1_crr, rot2_last, rot2_crr, rot2_sim

    rot1 = rot
    if rot1_last is None:
        rot1_last = initial
    delta1 = rot1* rot1_last.inv()
    rot1_last = rot1

    rot1_crr = ( delta1*rot1_crr )
    pose = rot1_crr.as_euler("xyz")


    rot2 = handedness_cor_system(rot)

    if rot2_last is None:
        rot2_last = handedness_cor_system(initial)
        # rot2_sim = R.from_euler('xyz',[0, 0,0 ])
    delta2 = (rot2) * (rot2_last).inv()
    delta2 = (delta2)
    rot2_last = rot2

    # rot2_sim_last = (rot2_sim)
    # rot2_sim = (delta2*rot2_sim)
    # delta2_sim = inverse_special(rot2_sim) * inverse_special(rot2_sim_last).inv()
    

    rot2_crr = ( delta2*rot2_crr )

    print(delta1.as_euler("xyz"), delta1.as_rotvec())    
    # assert change_cor_system(delta2).approx_equal(delta1)
    # assert rvec_filp(rot1_crr).approx_equal(inverse_euler(rot1_crr))
    # print("="*100)
    # print(rot2_crr.as_rotvec())
    rotv_swap = (rot2_crr).as_euler("xyz")
    # print(rot2_crr.as_rotvec())

    
    # rot_vector = rot.as_rotvec()
    # if debug:
    #     print(rot_vector)
    # rot_vector[0],rot_vector[1] = rot_vector[1],rot_vector[0]
    # if debug:
    #     print("-", rot_vector)
    # rot_rev = R.from_rotvec(rot_vector)
    # rotv_swap = rot_rev.as_euler("xyz")
    # pose2[0], pose2[1] = pose2[1], pose2[0]


    # show_plots(R.from_euler("xyz",pose2 ))
    for dim in range(3):
        # rr.log(f"normal/{names[dim]}", rr.Scalar(rot.as_euler("xyz")[dim]))
        rr.log(f"state/{names[dim]}", rr.Scalar(pose[dim]))
        rr.log(f"rotV_swp/{names[dim]}", rr.Scalar(rotv_swap[dim]))


def visualize_dataset(
) -> Path | None:
    mode ="local"
    spawn_local_viewer = mode 
    rr.init(f"episode_", spawn=spawn_local_viewer)

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
    
    initial = R.from_euler('xyz',[0.2,0.2 ,0.2 ])
    # for i in range(100):
    #     pnt(R.identity() ,i,0,True)
    #     # time.sleep(0.1)
    # for j in range(100):
    #     pnt(R.identity(),0,j)
    global rot1_last,rot1_crr, rot2_last, rot2_crr
    for i in range(100):
        pnt(initial,i,0)
            # time.sleep(0.1)
    
    rot1_last = None
    rot1_crr = R.from_euler('xyz',[0, 0,0 ])

    rot2_last = None
    rot2_crr = R.from_euler('xyz',[0, 0,0 ])
    for j in range(100):
        pnt(initial,0,j)
    




    # elif mode == "distant":
    #     # stop the process from exiting since it is serving the websocket connection
    #     try:
    #         while True:
    #             time.sleep(1)
    #     except KeyboardInterrupt:
    #         print("Ctrl-C received. Exiting.")


ax = plt.figure().add_subplot(projection="3d", proj_type="ortho")
ax.set(xlim=(-1.25, 5.25), ylim=(-1.25, 1.25), zlim=(-1.25, 1.25))

ax.set(xticks=range(-1, 8), yticks=[-1, 0, 1], zticks=[-1, 0, 1])

ax.set_aspect("equal", adjustable="box")

ax.figure.set_size_inches(6, 5)
_ = ax.annotate(

    "r0: Identity Rotation\n"

    "r1: Intrinsic Euler Rotation (ZYX)\n",


    xy=(0.6, 0.7), xycoords="axes fraction", ha="left"

)

def show_plots(r1):
    r0 = R.identity()
    

    plot_rotated_axes(ax, r0, name="r0", offset=(0, 0, 0))

    plot_rotated_axes(ax, r1, name="r1", offset=(3, 0, 0))






    plt.tight_layout()
    plt.show(block = True)


def main():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    kwargs = vars(args)
    # repo_id = kwargs.pop("repo_id")
    # root = kwargs.pop("root")
    # tolerance_s = kwargs.pop("tolerance_s")

    # logging.info("Loading dataset") #episodes=range(50),
    # dataset = LeRobotDataset(repo_id,  root=root, tolerance_s=tolerance_s)
    # eule = [0.2, 1.3, -1.4]
    eule = [-1.4, 1.3, 0.2]
    r = R.from_euler("xyz", eule)
    print(eule, r.as_quat())

    # r.approx_equal()

    visualize_dataset(**vars(args))
    # print(R.from_euler('xyz',[0,0,np.pi/2 ]).as_matrix())

    rt = R.from_euler('xyz',[0,np.pi*5/6,0 ])*R.from_euler('xyz',[np.pi,0, np.pi ]) 
    print(rt.as_euler("xyz"))


    

   

if __name__ == "__main__":
    main()

