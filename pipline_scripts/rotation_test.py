from scipy.spatial.transform import Rotation as R
import numpy as np
from numpy.linalg import inv



# what is now happening because of the left-handed coordinate system
def bad_change_cor_system(r, inverse=False):
    rvec = r.as_rotvec()
    rvec[0],rvec[1] = rvec[0], -rvec[1]
    
    return R.from_rotvec(rvec)


def bad_change_cor_system(r: R, inverse=False):
    # Mirror across Y-axis (e.g., left-handed system)
    M = np.diag([1, -1, 1])  # Reflect Y
    if inverse:
        # Undo the coordinate change
        mr = M @ r.as_matrix() @ M
    else:
        # Apply the coordinate change
        mr = M @ r.as_matrix() @ M
    return R.from_matrix(mr)

def bad_change_cor_system(r: R, inverse=False):
    M =  np.matrix('0 1 0; 1 0 0 ; 0 0 1')
    mr = inv(M) @ r.as_matrix() @ (M)
    return R.from_matrix(mr)


def change_cor_system(r, inverse=False):
    rvec = r.as_rotvec()
    if inverse:
        rvec[0],rvec[1] = rvec[1], -rvec[0]
    else:
        rvec[0],rvec[1] = -rvec[1], rvec[0]
    return R.from_rotvec(rvec)



# def apply_action(initial_r, action_r):
#     return  (action_r)* (initial_r)
def apply_action(initial_r, action_r):
    mr = (action_r).as_matrix() @ (initial_r).as_matrix()
    return  R.from_matrix(mr)

# is meant to simulate controlling robot using delta actions
def apply_actions(initial_rvec, actions_rvec):

    crr_rvec = initial_rvec
    for action_rvec in actions_rvec:
        crr_R = apply_action(crr_rvec, action_rvec)
        crr_rvec = R.as_rotvec(crr_R)
    return crr_R


def test1(seed):
    initial_rotation = R.random(rng=seed)
    action_rotation = R.random(rng=(seed+100))
    # assert initial_rotation.approx_equal(action_rotation)

    r1 = apply_action(initial_rotation, action_rotation)

    left_handed_initial_rvect = bad_change_cor_system(initial_rotation)
    left_handed_action = bad_change_cor_system(action_rotation)
    r2 = apply_action(left_handed_initial_rvect, left_handed_action)

    r2 = bad_change_cor_system(r2, inverse=True)
    
    assert r1.approx_equal(r2)
    print("test1 pass")

# def test2(seed):
#     initial_rotation_rvec = R.identity().as_rotvec()
#     actions_rotation_rvec = R.random(num=10, rng=seed).as_rotvec()


#     r1 = apply_actions(initial_rotation_rvec, actions_rotation_rvec)

#     left_handed_initial_rvec = change_cor_system(initial_rotation_rvec)
#     left_handed_actions_rvec =  [change_cor_system(action) for action in actions_rotation_rvec]
#     r2= apply_actions(left_handed_initial_rvec, left_handed_actions_rvec)

#     r2 = R.from_rotvec( change_cor_system(r2.as_rotvec(), inverse=True) )


#     assert r1.approx_equal(r2)
#     print("test2 pass")

def test3(seed):
    initial_rotation_rvec = R.identity()
    actions_rotation_rvec = R.random(num=5, rng=seed)


    

    r1 = initial_rotation_rvec
    r2 = bad_change_cor_system(initial_rotation_rvec)
    for i,act in enumerate(actions_rotation_rvec):
        r1 = apply_action(r1, act)

        r2= apply_action(r2, bad_change_cor_system(act))

        if not r1.approx_equal(bad_change_cor_system(r2)): #, atol=0.1
            print(i)
        else:
            print(f"ind={i} pass")
        # if not r2.approx_equal(bad_change_cor_system(r1), atol=0.1):
        #     print("rev",i)    
        # assert r1.approx_equal(bad_change_cor_system(r2), atol=0.1)
        # 

        r1 =bad_change_cor_system(r2)

    print("test3 pass")


def test4(seed):
    initial_rotation_rvec = R.identity().as_rotvec()
    actions_rotation_rvec = R.random(num=2, rng=seed).as_rotvec()
    actions_rotation_rvec = actions_rotation_rvec[1:]


    r1 = apply_actions(initial_rotation_rvec, actions_rotation_rvec)

    left_handed_initial_rvec = bad_change_cor_system(initial_rotation_rvec)
    left_handed_actions_rvec =  [bad_change_cor_system(action) for action in actions_rotation_rvec]
    r2= apply_actions(left_handed_initial_rvec, left_handed_actions_rvec)

    r2 = R.from_rotvec( bad_change_cor_system(r2.as_rotvec(), inverse=True) )


    assert r1.approx_equal(r2, atol=0.1)
    print("test3 pass")

def main(n):
    test1(n)
    # test2(n)
    test3(n)

if __name__ == "__main__":
    for i in range(10):
        main(i)
