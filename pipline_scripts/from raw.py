from lerobot.common.datasets.push_dataset_to_hub.utils import check_repo_id
from lbot.umi_zarr_format import from_raw_to_lerobot_format

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset


import click
import os
import shutil
from pathlib import Path

@click.command()
@click.option('-i', 'input_dir',  required=True,)
@click.option('-l', '--local_dir', required=True, help='temp folder for storing lrdataset localy')
def main(input_dir, local_dir):

# from_raw_to_lerobot_format(
#     raw_dir: Path,
#     videos_dir: Path,
#     fps: int | None = None,
#     video: bool = True,
#     episodes: list[int] | None = None,
#     encoding: dict | None = None,
# ):
    print(input_dir)
    print(local_dir)
    input_path = Path(os.path.expanduser(input_dir)).absolute()
    print(input_path)
    
    repo_id="eyefly2/test"
    check_repo_id(repo_id)

    # TODO:
    force_override = True

    local_dir = Path(os.path.expanduser(local_dir)).absolute()
    if local_dir.exists():
        if force_override:
            shutil.rmtree(local_dir)
        elif not resume:
            raise ValueError(f"`local_dir` already exists ({local_dir}). Use `--force-override 1`.")
    meta_data_dir = local_dir / "meta_data"
    videos_dir = local_dir / "videos"

    zarr_path = input_path / "cup_in_the_wild.zarr"
    print(zarr_path)

    # TODO add/use video: bool = true to minimize amount of RAM/mem used during conversion
    use_video = True
    hf_dataset, episode_data_index, info = from_raw_to_lerobot_format(input_path,videos_dir, video=use_video)

    
    print()
    print(hf_dataset)
    print()
    print(episode_data_index)
    print()
    print(info)

    print(hf_dataset.features)
    print(type(hf_dataset.features.to_dict()))

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps = info["fps"], #might be wrong
        features=hf_dataset.features.to_dict(),
        use_videos=use_video,

    )
    dataset.hf_dataset = hf_dataset

    dataset.push_to_hub(private=True, upload_large_folder=True)


    # lerobot_dataset = LeRobotDataset.from_preloaded(
    #     repo_id=repo_id,
    #     hf_dataset=hf_dataset,
    #     episode_data_index=episode_data_index,
    #     info=info,
    #     videos_dir=videos_dir,
    # )

    # dataset.push_to_hub(private=True, upload_large_folder=True)



    #  ==================================================

    # stats = compute_stats(lerobot_dataset, batch_size, num_workers)

    # if local_dir:
    #     hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
    #     hf_dataset.save_to_disk(str(local_dir / "train"))

    # if push_to_hub or local_dir:
    #     # mandatory for upload
    #     save_meta_data(info, stats, episode_data_index, meta_data_dir)

    # if push_to_hub:
    #     hf_dataset.push_to_hub(repo_id, revision="main")
    #     push_meta_data_to_hub(repo_id, meta_data_dir, revision="main")
    #     push_dataset_card_to_hub(repo_id, revision="main")
    #     if video:
    #         push_videos_to_hub(repo_id, videos_dir, revision="main")
    #     create_branch(repo_id, repo_type="dataset", branch=CODEBASE_VERSION)



# %%
if __name__ == "__main__":
    main()