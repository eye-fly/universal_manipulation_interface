from scipy.spatial.transform import Rotation as R
import numpy as np

# function for alining rotation from umi frame to robot frame
def offset_rot(umi_pose):
    translation = np.array([-0.7, -0.25, 0.0])

    umi_pose[:3] = umi_pose[:3]+translation
    # offset = [-0.11115846,  1.19299307, -0.59110642]
    # offset_rot = R.from_euler("xyz", offset)


    # ret_pose = umi_pose.copy()

    # umi_rot = R.from_euler("xyz", umi_pose[3:])

    # ret_pose[3:] = (umi_rot*offset_rot).as_euler("xyz")
    return umi_pose