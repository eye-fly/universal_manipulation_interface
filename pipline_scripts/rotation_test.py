from scipy.spatial.transform import Rotation as R
import numpy as np



# what is now happening because of the left-handed coordinate system
def bad_change_cor_system(rvec, inverse=False):
    rvec[0],rvec[1] = rvec[1], rvec[0]
    return rvec

def change_cor_system(rvec, inverse=False):
    if inverse:
        rvec[0],rvec[1] = rvec[1], -rvec[0]
    else:
        rvec[0],rvec[1] = -rvec[1], rvec[0]
    return rvec



def apply_action(initial_rvec, action_rvec):
    return  R.from_rotvec(action_rvec)* R.from_rotvec(initial_rvec)

# is meant to simulate controlling robot using delta actions
def apply_actions(initial_rvec, actions_rvec):

    crr_rvec = initial_rvec
    for action_rvec in actions_rvec:
        crr_R = apply_action(crr_rvec, action_rvec)
        crr_rvec = R.as_rotvec(crr_R)
    return crr_R


def test1(seed):
    initial_rotation = R.random(rng=seed).as_rotvec()
    action_rotation = R.random(rng=seed).as_rotvec()

    r1 = apply_action(initial_rotation, action_rotation)

    left_handed_initial_rvect = bad_change_cor_system(initial_rotation)
    left_handed_action = bad_change_cor_system(action_rotation)
    r2 = apply_action(left_handed_initial_rvect, left_handed_action)

    r2 = R.from_rotvec( bad_change_cor_system(r2.as_rotvec(), inverse=True) )
    
    assert r1.approx_equal(r2)
    print("test1 pass")

def test2(seed):
    initial_rotation_rvec = R.identity().as_rotvec()
    actions_rotation_rvec = R.random(num=10, rng=seed).as_rotvec()


    r1 = apply_actions(initial_rotation_rvec, actions_rotation_rvec)

    left_handed_initial_rvec = change_cor_system(initial_rotation_rvec)
    left_handed_actions_rvec =  [change_cor_system(action) for action in actions_rotation_rvec]
    r2= apply_actions(left_handed_initial_rvec, left_handed_actions_rvec)

    r2 = R.from_rotvec( change_cor_system(r2.as_rotvec(), inverse=True) )


    assert r1.approx_equal(r2, atol=0.1)
    print("test2 pass")


def test3(seed):
    initial_rotation_rvec = R.identity().as_rotvec()
    actions_rotation_rvec = R.random(num=2, rng=seed).as_rotvec()


    r1 = apply_actions(initial_rotation_rvec, actions_rotation_rvec)

    left_handed_initial_rvec = bad_change_cor_system(initial_rotation_rvec)
    left_handed_actions_rvec =  [bad_change_cor_system(action) for action in actions_rotation_rvec]
    r2= apply_actions(left_handed_initial_rvec, left_handed_actions_rvec)

    r2 = R.from_rotvec( bad_change_cor_system(r2.as_rotvec(), inverse=True) )


    assert r1.approx_equal(r2, atol=0.1)
    print("test3 pass")

def main(n):
    test1(n)
    test2(n)
    test3(n)

if __name__ == "__main__":
    for i in range(1):
        main(i)
