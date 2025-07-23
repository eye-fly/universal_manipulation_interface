# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import json
import pathlib
import click
import zarr
import pickle
import numpy as np
import cv2
import av
import multiprocessing
import concurrent.futures
from tqdm import tqdm
import time
import torch
import threading
from scipy.spatial.transform import Rotation as R


from src.universal_manipulation_interface.umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter,
    get_image_transform, 
    draw_predefined_mask,
    inpaint_tag,
    get_mirror_crop_slices
)
from src.universal_manipulation_interface.diffusion_policy.common.replay_buffer import ReplayBuffer
from src.universal_manipulation_interface.diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl

from src.lbot.umi_zarr_format import from_raw_to_lerobot_format, umi_feats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import check_repo_id
register_codecs()




# %%
@click.command()
@click.argument('input', nargs=-1)
@click.option('-rp', '--repo_id', required=True, help='hf dataset name')
@click.option('-or', '--out_res', type=str, default='224,224')
@click.option('-of', '--out_fov', type=float, default=None)
# @click.option('-cl', '--compression_level', type=int, default=99)
@click.option('-nm', '--no_mirror', is_flag=True, default=False, help="Disable mirror observation by masking them out")
@click.option('-ms', '--mirror_swap', is_flag=True, default=False)
@click.option('-n', '--num_workers', type=int, default=None)
def main(input, repo_id, out_res, out_fov, 
         no_mirror, mirror_swap, num_workers):
    # if os.path.isfile(output):
    #     if click.confirm(f'Output file {output} exists! Overwrite?', abort=True):
    #         pass
        
    out_res = tuple(int(x) for x in out_res.split(','))

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    cv2.setNumThreads(1)
            
    fisheye_converter = None
    if out_fov is not None:
        intr_path = pathlib.Path(os.path.expanduser(ipath)).absolute().joinpath(
            'calibration',
            'gopro_intrinsics_2_7k.json'
        )
        opencv_intr_dict = parse_fisheye_intrinsics(json.load(intr_path.open('r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=out_res,
            out_fov=out_fov
        )
        
    # out_replay_buffer = ReplayBuffer.create_empty_zarr(
    #     storage=zarr.MemoryStore())


    def video_to_pic(mp4_path, tasks):
        pkl_path = os.path.join(os.path.dirname(mp4_path), 'tag_detection.pkl')
        tag_detection_results = pickle.load(open(pkl_path, 'rb'))
        
        
        curr_task_idx = 0
        img_array = []

        is_mirror = None
        if mirror_swap:
            ow, oh = out_res
            mirror_mask = np.ones((oh,ow,3),dtype=np.uint8)
            mirror_mask = draw_predefined_mask(
                mirror_mask, color=(0,0,0), mirror=True, gripper=False, finger=False)
            is_mirror = (mirror_mask[...,0] == 0)
        
        # from torchcodec.decoders._video_decoder import supported_devices
        # print(f"{torch.__version__=}")
        # print(f"{torch.cuda.is_available()=}")
        # print(f"{torch.cuda.get_device_properties(0)=}")
        # # print(torchcodec.cuda.is_available())
        # print()
        
        # device = "cuda"  # or e.g. "cuda" !
        # testtt =torch.rand(10).to(device)
        # print(testtt)
        # decoder= VideoDecoder(mp4_path, device=device)
        # container = decoder[: ]
        with av.open(mp4_path) as container:
            in_stream = container.streams.video[0]

        # print(container)
        # print(container.shape)
        # ih, iw = container.shape[2], container.shape[3]
            ih, iw = in_stream.height, in_stream.width
            resize_tf = get_image_transform(
                in_res=(iw, ih),
                out_res=out_res
            )

            # in_stream.thread_type = "AUTO"
            in_stream.thread_count = 1

            buffer_idx = 0
            # for frame_idx, frame in tqdm(enumerate(container), total=container.shape[0], leave=False):
            for frame_idx, frame in tqdm(enumerate(container.decode(in_stream)), total=in_stream.frames, leave=False):
                # if curr_task_idx >= len(tasks):
                #     # all tasks done
                #     break
                
                if frame_idx < tasks['frame_start']:
                    # current task not started
                    continue
                elif frame_idx < tasks['frame_end']:
                    # if frame_idx == tasks[curr_task_idx]['frame_start']:
                    #     buffer_idx = tasks[curr_task_idx]['buffer_start']
                    
                    # do current task
                    img = frame.to_ndarray(format='rgb24')
                    # if(device =="cuda" ):
                    #     frame = frame.device("cpu")
                    # img = frame.permute(1, 2, 0).cpu().numpy()

                    # inpaint tags
                    this_det = tag_detection_results[frame_idx]
                    all_corners = [x['corners'] for x in this_det['tag_dict'].values()]
                    for corners in all_corners:
                        img = inpaint_tag(img, corners)
                        
                    # mask out gripper
                    img = draw_predefined_mask(img, color=(0,0,0), 
                        mirror=no_mirror, gripper=True, finger=False)
                    # resize
                    if fisheye_converter is None:
                        img = resize_tf(img)
                    else:
                        img = fisheye_converter.forward(img)
                        
                    # handle mirror swap
                    if mirror_swap:
                        img[is_mirror] = img[:,::-1,:][is_mirror]
                        
                    # compress image
                    # print(img)
                    # print(img.shape)
                    img_array.append(img)
                    buffer_idx += 1
            return img_array
    

    
    # lerobot dataset
    use_video = True #TODO
    fps = 10 #TODO
    # task = "cup arrangment" #TODO
    tasks = ["front/back","left/right", "down/up","around x ccw", "around z to left", "around y to down" ] #TODO

    check_repo_id(repo_id)
    


    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps = fps,
        features=umi_feats(224,224), #TODO
        use_videos=use_video,

    )
    
    def euler_fix_delta(delta):
        # delta = euler2 - euler1
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        return delta

    mutex = threading.Lock()
    def process_whole_video(plan_episode, plan_nr):
        grippers = plan_episode['grippers']
        cameras = plan_episode['cameras']

        # TODO: currently onli MONOSETUP
        gripper = grippers[0]
        camera = cameras[0]

        eef_pose = gripper['tcp_pose']
        # eef_pos = eef_pose[...,:3]
        # eef_rot = eef_pose[...,3:]
        gripper_widths = gripper['gripper_width']
        # demo_start_pose = np.empty_like(eef_pose)
        # demo_start_pose[:] = gripper['demo_start_pose']
        # demo_end_pose = np.empty_like(eef_pose)
        # demo_end_pose[:] = gripper['demo_end_pose']
        
        video_path_rel = camera['video_path']
        video_path = demos_path.joinpath(video_path_rel).absolute()
        assert video_path.is_file()
        video_start, video_end = camera['video_start_end']

        start_time = time.time()
        video_arr = video_to_pic(video_path,{
                'frame_start': video_start,
                    'frame_end': video_end,
                    }) #TODO: alternatively use ffmpg /torch codec
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"video_to_pic took: {elapsed_time:.2f} seconds")

        print(eef_pose.shape)
        print(len(video_arr))
        # print(video_arr)

        
        with mutex:
            frame_n = gripper['tcp_pose'].shape[0]

            last_pose = None
            for frame_i in range(frame_n):
                frame = dict()
                frame["task"] = tasks[plan_nr% len(tasks)]

                crr_pose = eef_pose[frame_i].astype('float32')
                rot = crr_pose[3:]
                # crr_pose[3:] = R.from_rotvec([rot[2], rot[1], rot[0]]).as_euler('xyz') #  TOCHECK: x,y axis are/where swiched compared to simulation
                
                # rot = rot * R.from_euler('xyz',[0,0,np.pi/2 ])

                euler_yxz = R.from_rotvec(rot).as_euler('yxz') #  TOCHECK: x,y axis are/where swiched compared to simulatio
                crr_pose[3:] = [euler_yxz[1], euler_yxz[0], euler_yxz[2]]

                # print(crr_pose[3:])

                # # fix order of axis
                # # swap x with y axis
                # crr_pose[0], crr_pose[1] = crr_pose[1], crr_pose[0]
                # # reverse x, y
                crr_pose[0] = -crr_pose[0]
                crr_pose[1] = -crr_pose[1]

                # # reverse pitch and yaw
                # frame["observation.state.pose"][4] = -frame["observation.state.pose"][4]
                # frame["observation.state.pose"][5] = -frame["observation.state.pose"][5]
                # # swap pitch with yaw       



                frame["observation.state.pose"]  = crr_pose
                frame["action.gripper"] = np.array([gripper_widths[frame_i]], dtype='float32')
                
                frame["action.pose"] = np.zeros_like(crr_pose)
                if not (last_pose is None):
                    frame["action.pose"][:3] = crr_pose[:3] - last_pose[:3]

                    #totation form last_pose to current 
                    delta_rot = R.from_euler( "xyz", crr_pose[3:]) * R.from_euler( "xyz", last_pose[3:]).inv()
                    frame["action.pose"][3:] = delta_rot.as_euler("xyz")



         

                frame["observation.images"] = video_arr[frame_i]
                tpf = 1.0/fps
                frame["timestamps"] = np.array([tpf*frame_i]).astype('float32')

                dataset.add_frame(frame)
                last_pose = crr_pose
            dataset.save_episode()


    # dump lowdim data to replay buffer
    # generate argumnet for videos
    n_grippers = None
    n_cameras = None
    # buffer_start = 0
    # all_videos = set()
    vid_args = list()
    videos_used = 0
    for ipath in input:
        ipath = pathlib.Path(os.path.expanduser(ipath)).absolute()
        demos_path = ipath.joinpath('demos')
        plan_path = ipath.joinpath('dataset_plan.pkl')
        if not plan_path.is_file():
            print(f"Skipping {ipath.name}: no dataset_plan.pkl")
            continue
        
        plan = pickle.load(plan_path.open('rb'))
        
        
        plan_nr = 0
        with tqdm(total=len(plan)) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = set()
                for plan_episode in plan:
                    if len(futures) >= num_workers:
                        # limit number of inflight tasks
                        completed, futures = concurrent.futures.wait(futures, 
                            return_when=concurrent.futures.FIRST_COMPLETED)
                        pbar.update(len(completed))

                    grippers = plan_episode['grippers']
                    # check that all episodes have the same number of grippers 
                    if n_grippers is None:
                        n_grippers = len(grippers)
                    else:
                        assert n_grippers == len(grippers)
                        
                    cameras = plan_episode['cameras']
                    if n_cameras is None:
                        n_cameras = len(cameras)
                    else:
                        assert n_cameras == len(cameras)
                    
 #                   process_whole_video(plan_episode, plan_nr) #FOR debug only
                    futures.add(executor.submit(process_whole_video, plan_episode, plan_nr))
                    videos_used+=1
                    plan_nr +=1
                    # process_whole_video(plan_episode)#==========================

                completed, futures = concurrent.futures.wait(futures)
                pbar.update(len(completed))
            
        # videos_dict = defaultdict(list)
        
            


            
            # out_replay_buffer.add_episode(data=episode_data, compressors=None)
            
            # aggregate video gen aguments
        #     n_frames = None
        #     for cam_id, camera in enumerate(cameras):
        #         
                
        #         video_start, video_end = camera['video_start_end']
        #         if n_frames is None:
        #             n_frames = video_end - video_start
        #         else:
        #             assert n_frames == (video_end - video_start)
                
        #         videos_dict[str(video_path)].append({
        #             'camera_idx': cam_id,
        #             'frame_start': video_start,
        #             'frame_end': video_end,
        #             'buffer_start': buffer_start
        #         })
        #     buffer_start += n_frames
        
        # vid_args.extend(videos_dict.items())
        # all_videos.update(videos_dict.keys())
        
    
    print(f"{videos_used} videos used in total!")
    dataset.push_to_hub(private=True)
    

    # print(episode_data)
    # print()
    
    
    # # dump images
    # img_compressor = JpegXl(level=compression_level, numthreads=1)
    # for cam_id in range(n_cameras):
    #     name = f'camera{cam_id}_rgb'
    #     _ = out_replay_buffer.data.require_dataset(
    #         name=name,
    #         shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_res + (3,),
    #         chunks=(1,) + out_res + (3,),
    #         compressor=img_compressor,
    #         dtype=np.uint8
    #     )

   
                #     if (frame_idx + 1) == tasks[curr_task_idx]['frame_end']:
                #         # current task done, advance
                #         curr_task_idx += 1
                # else:
                #     assert False
                    
    # with tqdm(total=len(vid_args)) as pbar:
    #     # one chunk per thread, therefore no synchronization needed
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    #         futures = set()
    #         for mp4_path, tasks in vid_args:
    #             if len(futures) >= num_workers:
    #                 # limit number of inflight tasks
    #                 completed, futures = concurrent.futures.wait(futures, 
    #                     return_when=concurrent.futures.FIRST_COMPLETED)
    #                 pbar.update(len(completed))

    #             futures.add(executor.submit(video_to_zarr, 
    #                 out_replay_buffer, mp4_path, tasks))

    #         completed, futures = concurrent.futures.wait(futures)
    #         pbar.update(len(completed))

    # print([x.result() for x in completed])

    # # dump to disk
    # print(f"Saving ReplayBuffer to {output}")
    # with zarr.ZipStore(output, mode='w') as zip_store:
    #     out_replay_buffer.save_to_store(
    #         store=zip_store
    #     )
    # print(f"Done! {len(all_videos)} videos used in total!")

# %%
if __name__ == "__main__":
    main()

