import os
from goal_segment.goal_segment import goal_segment  
from goal_segment.crop_video import crop_video_with_coordinates
from goal_segment.keeper_posing import keeper_posing
from optical_flow.add_optical_flow_to_dataset import add_optical_flow_to_dataset
from base import INPUT_VIDEO_FOLDER, GOAL_OUTPUT_FOLDER, SPLIT_FRAME, INPUT_FRAME, OUTPUT_FRAME, CROPPED_VIDEO_FOLDER

def generate_dataset_from_video():
    total = 0
    print("データセットの作成を開始します。")
    # # フォルダ内の全ファイルを取得
    # video_files = sorted(os.listdir(INPUT_VIDEO_FOLDER))
    # for input_video_name in video_files:
    #     goal_output_file_path = goal_segment(input_video_name)
    #     crop_video_with_coordinates(input_video_name, goal_output_file_path)
    # for input_video_name in video_files:
    #     if os.path.exists(os.path.join(CROPPED_VIDEO_FOLDER, "cropped_" + input_video_name)):
    #         total += 1
    #         keeper_posing(input_video_name, os.path.join(GOAL_OUTPUT_FOLDER, input_video_name + "_goal.json"), SPLIT_FRAME, INPUT_FRAME, OUTPUT_FRAME)
    add_optical_flow_to_dataset()
    print(f"全てで{total}件のデータセットが生成されました。")

if __name__ == "__main__": 
    generate_dataset_from_video()


# srun -p p -t 60:00 --gres=gpu:1 --pty poetry run python src/making_datasets/generate_dataset.py