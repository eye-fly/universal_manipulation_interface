install
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
sudo apt-get install ffmpeg


mamba env create -f universal_manipulation_interface/conda_environment.yaml



run pipline
conda run -n umi python src/universal_manipulation_interface/run_slam_pipeline.py /home/pzero/nomagic/example/example_demo_session

uv run pipline_scripts/07_generate_el_dataset.py example_demo_session/

uv run pipline_scripts/visualize.py --repo-id eyefly2/test --episode-index 0




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

        