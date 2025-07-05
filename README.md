install
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
sudo apt-get install ffmpeg


mamba env create -f universal_manipulation_interface/conda_environment.yaml



run pipline
conda run -n umi python universal_manipulation_interface/run_slam_pipeline.py /home/pzero/nomagic/example/example_demo_session


