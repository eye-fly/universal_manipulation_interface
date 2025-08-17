from scipy.spatial.transform import Rotation as R

# function for alining rotation from umi frame to robot frame
def offset_rot(umi_pose):
    offset = [-0.11115846,  1.19299307, -0.59110642]
    offset_rot = R.from_euler("xyz", offset)


    ret_pose = umi_pose.copy()

    umi_rot = R.from_euler("xyz", umi_pose[3:])

    ret_pose[3:] = (umi_rot*offset_rot).as_euler("xyz")
    return ret_pose