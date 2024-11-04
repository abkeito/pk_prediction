import cv2
import numpy as np
import json
from base import CROPPED_VIDEO_FOLDER, INPUT_SPLIT_FOLDER, POSE_OUTPUT_FOLDER, GOAL_OUTPUT_FOLDER, OPTICAL_FOLDER, DATASET_FOLDER
import os
from optical_flow import optical_flow

# JSONファイルを読み込む関数
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 距離計算関数
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# 平均を計算する関数
def calculate_average_coordinates(optical_data, keeper_pose_data, threshold=0.5):

    for frame_id in range(min(len(optical_data), len(keeper_pose_data))):
        average_coords = {}
        keeper_pose = keeper_pose_data[frame_id]["keeper-pose"]
        before = optical_data[frame_id]["before"]
        after = optical_data[frame_id]["after"]

        if keeper_pose is not None:
            for body_part, part_coords in keeper_pose.items():
                close_indices = []
                for index, coord in enumerate(before):
                    distance = calculate_distance(coord, part_coords)
                    if distance < threshold:
                        close_indices.append(index)

                if close_indices:
                    # インデックスに基づいて座標を平均化
                    average_coords[body_part + "-move"] = np.mean([np.array(after[i]) - np.array(before[i]) for i in close_indices], axis=0).tolist()

                else:
                    average_coords[body_part  + "-move"] = [0, 0]  # 該当する座標がない場合

            keeper_pose_data[frame_id]["keeper-pose"] = keeper_pose_data[frame_id]["keeper-pose"] | average_coords

    return keeper_pose_data


# 主な処理
if __name__ == "__main__":
    optical_flow()
    # JSONファイルのパス
    filenames = os.listdir(OPTICAL_FOLDER)

    for filename in filenames:
        filenumber = filename.split("_")[0]

        optical_json_path = os.path.join(OPTICAL_FOLDER, filename)
        posing_json_path = os.path.join(POSE_OUTPUT_FOLDER, filenumber + ".mp4_pose.json")
        output_path = os.path.join(DATASET_FOLDER, filenumber + "_dataset.json")

        # JSONデータの読み込み
        optical_data = load_json(optical_json_path)
        keeper_pose_data = load_json(posing_json_path)

        # beforeの座標リストとkeeperのポーズデータを取得

        # 平均座標の計算
        keeper_pose_data_new = calculate_average_coordinates(optical_data, keeper_pose_data)

        # JSONファイルにオプティカルフローの座標を保存
        with open(output_path, 'w') as json_file:
            json.dump(keeper_pose_data, json_file, indent=4)

        print(f"New Dataset saved to {output_path}")

# srun -p p -t 60:00 --gres=gpu:1 --pty poetry run python src/making_datasets/optical_flow/generate_new_dataset.py