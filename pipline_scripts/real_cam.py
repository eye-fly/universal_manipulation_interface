import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)



from typing import Optional, List
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import time
import shutil
import math
# from multiprocessing.managers import SharedMemoryManager
# from src.universal_manipulation_interface.umi.real_world.rtde_interpolation_controller import RTDEInterpolationController
# from src.universal_manipulation_interface.umi.real_world.wsg_controller import WSGController
# from src.universal_manipulation_interface.umi.real_world.franka_interpolation_controller import FrankaInterpolationController
from src.universal_manipulation_interface.umi.real_world.multi_uvc_camera import MultiUvcCamera, VideoRecorder
# from src.universal_manipulation_interface.diffusion_policy.common.timestamp_accumulator import (
#     TimestampActionAccumulator,
#     ObsAccumulator
# )
# from src.universal_manipulation_interface.umi.common.cv_util import draw_predefined_mask
# from src.universal_manipulation_interface.umi.real_world.multi_camera_visualizer import MultiCameraVisualizer
# from src.universal_manipulation_interface.diffusion_policy.common.replay_buffer import ReplayBuffer
# from src.universal_manipulation_interface.diffusion_policy.common.cv2_util import (
#     get_image_transform, optimal_row_cols)
from src.universal_manipulation_interface.umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
# from src.universal_manipulation_interface.umi.common.pose_util import pose_to_pos_rot
# from src.universal_manipulation_interface.umi.common.interpolation_util import get_interp1d, PoseInterpolator


reset_all_elgato_devices()

# Wait for all v4l cameras to be back online
time.sleep(0.1)
v4l_paths = get_sorted_v4l_paths()

print(v4l_paths)

# shm_manager = SharedMemoryManager()
# shm_manager.start()

res = (1920, 1080)

# def vis_tf(data, input_res=res):
#     img = data['color']
#     f = get_image_transform(
#         input_res=input_res,
#         output_res=(rw,rh),
#         bgr_to_rgb=False
#     )
#     img = f(img)
#     data['color'] = img
#     return data
# vis_transform.append(vis_tf)


with MultiUvcCamera(
    dev_video_paths=v4l_paths,
    # shm_manager=shm_manager,
    resolution=res,
    # capture_fps=capture_fps,
    # # send every frame immediately after arrival
    # # ignores put_fps
    put_downsample=False,
    # get_max_k=max_obs_buffer_size,
    # receive_latency=camera_obs_latency,
    # cap_buffer_size=cap_buffer_size,
    # transform=transform,
    # vis_transform=vis_transform,
    # video_recorder=video_recorder,
    verbose=True
) as camera:

    multi_cam_vis = None
    # if enable_multi_cam_vis:
    #     multi_cam_vis = MultiCameraVisualizer(
    #         camera=camera,
    #         row=row,
    #         col=col,
    #         rgb_to_bgr=False
    #     )

    # camera.start()
    # camera.start_wait()
    # if multi_cam_vis is not None:
    #     multi_cam_vis.start_wait()

    for i in range(10):
        camera_data = camera.get()

    print(camera.is_ready)

    plt.imshow(camera_data[0]['color'])
    plt.axis('off')
    plt.show()

    # vis = camera.get_vis()
    camera.stop_recording()
    exit(0)
    # print(vis)
    # camera.stop_wait()
    # if multi_cam_vis is not None:
    #     multi_cam_vis.stop_wait()


