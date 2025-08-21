TOD0: swich to a gopro12 banch for  submodule :
  git config -f .gitmodules submodule.src/universal_manipulation_interface.branch gopro12
  git submodule update --remote


install
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
sudo apt-get install ffmpeg



docker build -t umi .

1. 
VIDEO_FULLPATH="$(realpath ~/Wideo/vid2)"
docker run --rm -it -v  $VIDEO_FULLPATH:$VIDEO_FULLPATH \
  -e VIDEO_FULLPATH="$VIDEO_FULLPATH" -e SLAM_SET_PATH="$SLAM_SET_PATH" \
  -v /usr/bin/docker:/usr/bin/docker -v /var/run/docker.sock:/var/run/docker.sock \
  umi
2. conda run -n umi python /workspace/UMI/src/universal_manipulation_interface/run_slam_pipeline.py $VIDEO_FULLPATH






mamba env create -f /workspace/UMI/src/universal_manipulation_interface/conda_environment.yaml -y


conda run -n umi python src/universal_manipulation_interface/run_slam_pipeline.py ~videos/

run pipline
conda run -n umi /home/pzero/miniforge3/envs/umi/bin/python src/universal_manipulation_interface/run_slam_pipeline.py /home/pzero/nomagic/example/videos/
(last timeworked::: conda run -n umi /home/pzero/miniforge3/envs/umi/bin/python universal_manipulation_interface/run_slam_pipeline.py /home/pzero/nomagic/example/videos/) - probobly because .venv from uv and ussing diffrent python ect mayby runging inside /src folder solves the problem

uv run pipline_scripts/07_generate_el_dataset.py example_demo_session/

uv run pipline_scripts/visualize.py --repo-id eyefly2/test --episode-index 0


-p 9876:9876 

docker run --rm -it -v  /home/jacek-muszynski/Videos:/home/jacek-muszynski/Videos     -e HF_TOKEN=""     -v /usr/bin/docker:/usr/bin/docker -v /var/run/docker.sock:/var/run/docker.sock     --privileged --network=host -e NVIDIA_DRIVER_CAPABILITIES=all -e DISPLAY=$DISPLAY   --gpus all  umi

python pipline_scripts/visualize.py --mode distant --ws-port 9876 --repo-id eyefly2/robot --episode-index 0 

python pipline_scripts/visualize.py --repo-id eyefly2/robot --episode-index 0 



uv run pipline_scripts/zarr_to_lerobot.py -rp eyefly2/cup -p /home/jacek-muszynski/Videos/cup_in_the_wild.zarr


=-------------------------
bash run_docker_pipline.sh -n eyefly2/nomagic ~/Videos/nomagic/
-=-----------------------------------

in dataset:
    x,y,z, rol, pitch, yaw

    x -- forward(+) - backward(-)
    y -- left(+) - right(-)
    z -- up(+) - down(-)
        
    clockwise means "looking" thourward + of a axis
    roll[around x] -- clockwise(+) pi/2 is 90deg
    pitch[around y] -- clockwise(+)
    yaw[around z] -- counterclockwise(+)
    

in nomagic

  x  right(-) left(+)
  y  forward(-) bacward(+)
  z up(+)

  roll [arund y] clockwise(+)
  pitch [around x] clockwise(+)
  yaw [around z] clockwise(+)

only left ot reverse direction of yaw



dataset with "Working [after changing xyz to yxz]" -- eyefly2/robot_basis || changing using change of basis(pi rot azround z)

"compatible" with current but need reverse of direction on pitch --  eyefly2/robot_unchanged_swaped_roll || changing using rotvect


robot_interface cantron and recording data at a same time



notes:
1. for slamp piline and current gripper:
    print("############# 06_generate_dataset_plan ###########")
            script_path = script_dir.joinpath("06_generate_dataset_plan.py")
            assert script_path.is_file()
            cmd = [
                'python', str(script_path),
                '--input', str(session),
                '-nz',  '0.0543', <<=== nomina z offset for some reazon is defferent than defoul one (maybe 
                because different goopr?)
and same for script_path = script_dir.joinpath('calibrate_gripper_range.py') [inside scripts_slam_pipline/05_run_cal,,,.py ]

        
        24 start


TCP(tool center point) in remote,, how to set?




=================================================================================================
/home/pzero/nomagic/example/src/universal_manipulation_interface/slam/gopro12_black_maxlens_fisheye_setting_v1.yaml




./Examples/Monocular-Inertial/gopro_slam -i /home/pzero/nomagic/example/videos/demos/demo_C3501326231484_2025.07.11_14.52.16.299483/raw_video.mp4 -j /home/pzero/nomagic/example/videos/demos/demo_C3501326231484_2025.07.11_14.52.16.299483/imu_data.json -g --mask_img /home/pzero/nomagic/example/videos/demos/demo_C3501326231484_2025.07.11_14.51.21.778350/slam_mask.png -s /home/pzero/nomagic/example/src/universal_manipulation_interface/slam/gopro12_black_maxlens_fisheye_setting_v1.yaml -v Vocabulary/ORBvoc.txt --load_map /home/pzero/nomagic/example/videos/demos/mapping/map_atlas.osa




./Examples/Monocular-Inertial/gopro_slam -i /home/pzero/nomagic/example/videos/demos/demo_C3501326231484_2025.07.11_14.52.03.853717/raw_video.mp4 -j /home/pzero/nomagic/example/videos/demos/demo_C3501326231484_2025.07.11_14.52.03.853717/imu_data.json -g --mask_img /home/pzero/nomagic/example/videos/demos/demo_C3501326231484_2025.07.11_14.51.21.778350/slam_mask.png -s /home/pzero/nomagic/example/src/universal_manipulation_interface/slam/gopro12_black_maxlens_fisheye_setting_v1.yaml -v Vocabulary/ORBvoc.txt --load_map /home/pzero/nomagic/example/videos/demos/mapping/worse_map_atlas.osa



/home/pzero/nomagic/example/videos/demos/mapping/better_map_atlas.osa


./Examples/Monocular-Inertial/gopro_slam -i /home/pzero/nomagic/example/videos/demos/mapping/raw_video.mp4 -j /home/pzero/nomagic/example/videos/demos/mapping/imu_data.json -g --mask_img /home/pzero/nomagic/example/videos/demos/demo_C3501326231484_2025.07.11_14.51.21.778350/slam_mask.png -s /home/pzero/nomagic/example/src/universal_manipulation_interface/slam/gopro12_black_maxlens_fisheye_setting_v1.yaml -v Vocabulary/ORBvoc.txt --save_map /home/pzero/nomagic/example/videos/demos/mapping/worse_map_atlas.osa




docker run --rm --volume /VideoIn/vid1/demos/mapping:/data chicheng/openicc:latest node /OpenImuCameraCalibrator/javascript/extract_metadata_single.js /data/raw_video.mp4 /data/imu_data.json



scp -P 6022 -r ~/Videos/test jmuszynski@nomagiclab.mimuw.edu.pl:~/Wideo/vid1

cat /home/jmuszynski/Wideo/vid2/demos/mapping/slam_stderr.txt