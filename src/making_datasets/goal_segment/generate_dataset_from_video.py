import os
from goal_segment import goal_segment
from crop_video import crop_video_with_coordinates
from keeper_posing import keeper_posing
from base import GOAL_SEGMENT_MODEL, INPUT_VIDEO_FOLDER, GOAL_OUTPUT_FOLDER, SPLIT_FRAME, INPUT_FRAME, OUTPUT_FRAME

def generate_dataset_from_video():
    print("データセットを作成します。")
    # フォルダ内の全ファイルを取得
    video_files = os.listdir(INPUT_VIDEO_FOLDER)
    for input_video_name in video_files:
        goal_output_file_path = goal_segment(input_video_name)
        crop_video_with_coordinates(input_video_name, goal_output_file_path)
        keeper_posing(input_video_name, goal_output_file_path, SPLIT_FRAME, INPUT_FRAME, OUTPUT_FRAME)

if __name__ == "__main__": 
    generate_dataset_from_video()

# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/making_datasets/goal_segment/generate_dataset_from_video.py