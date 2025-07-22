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

  roll [arund z] clockwise(-)
  pitch [around x] clockwise(+)
  yaw [around z] clockwise(+)

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





=================================================================================================
/home/pzero/nomagic/example/src/universal_manipulation_interface/slam/gopro12_black_maxlens_fisheye_setting_v1.yaml




./Examples/Monocular-Inertial/gopro_slam -i /home/pzero/nomagic/example/videos/demos/demo_C3501326231484_2025.07.11_14.52.16.299483/raw_video.mp4 -j /home/pzero/nomagic/example/videos/demos/demo_C3501326231484_2025.07.11_14.52.16.299483/imu_data.json -g --mask_img /home/pzero/nomagic/example/videos/demos/demo_C3501326231484_2025.07.11_14.51.21.778350/slam_mask.png -s /home/pzero/nomagic/example/src/universal_manipulation_interface/slam/gopro12_black_maxlens_fisheye_setting_v1.yaml -v Vocabulary/ORBvoc.txt --load_map /home/pzero/nomagic/example/videos/demos/mapping/map_atlas.osa




./Examples/Monocular-Inertial/gopro_slam -i /home/pzero/nomagic/example/videos/demos/demo_C3501326231484_2025.07.11_14.52.03.853717/raw_video.mp4 -j /home/pzero/nomagic/example/videos/demos/demo_C3501326231484_2025.07.11_14.52.03.853717/imu_data.json -g --mask_img /home/pzero/nomagic/example/videos/demos/demo_C3501326231484_2025.07.11_14.51.21.778350/slam_mask.png -s /home/pzero/nomagic/example/src/universal_manipulation_interface/slam/gopro12_black_maxlens_fisheye_setting_v1.yaml -v Vocabulary/ORBvoc.txt --load_map /home/pzero/nomagic/example/videos/demos/mapping/worse_map_atlas.osa



/home/pzero/nomagic/example/videos/demos/mapping/better_map_atlas.osa


./Examples/Monocular-Inertial/gopro_slam -i /home/pzero/nomagic/example/videos/demos/mapping/raw_video.mp4 -j /home/pzero/nomagic/example/videos/demos/mapping/imu_data.json -g --mask_img /home/pzero/nomagic/example/videos/demos/demo_C3501326231484_2025.07.11_14.51.21.778350/slam_mask.png -s /home/pzero/nomagic/example/src/universal_manipulation_interface/slam/gopro12_black_maxlens_fisheye_setting_v1.yaml -v Vocabulary/ORBvoc.txt --save_map /home/pzero/nomagic/example/videos/demos/mapping/worse_map_atlas.osa




docker run --rm --volume /VideoIn/vid1/demos/mapping:/data chicheng/openicc:latest node /OpenImuCameraCalibrator/javascript/extract_metadata_single.js /data/raw_video.mp4 /data/imu_data.json



scp -P 6022 -r ~/Videos/test jmuszynski@nomagiclab.mimuw.edu.pl:~/Wideo/vid1

cat /home/jmuszynski/Wideo/vid2/demos/mapping/slam_stderr.txt