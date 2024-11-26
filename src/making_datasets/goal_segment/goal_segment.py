import json
import numpy as np
from ultralytics import SAM, YOLO
import os
import torch
from base import GOAL_SEGMENT_MODEL, INPUT_VIDEO_FOLDER, GOAL_OUTPUT_FOLDER

# ビデオを受け取ってゴールの形をjson形式で持ってくる関数
# 入力：ビデオファイルの名前
# 出力：GOAL_OUTPUT_FOLDER 内に json形式が作られる
def goal_segment(input_video_name: str):
    # モデルを読み込む
    model = YOLO(GOAL_SEGMENT_MODEL)
    input_path = os.path.join(INPUT_VIDEO_FOLDER, input_video_name)
    goal_output_file_path = os.path.join(GOAL_OUTPUT_FOLDER, input_video_name + "_goal.json")

    # モデルの状況などの確認
    print(f"Goal segmentation started for {input_video_name}")
    print(torch.version.cuda)
    print(torch.__version__)
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    print(device)

    # モデルの情報を表示
    model.info()
    # モデルの予測を取得
    results = model(input_path, save=True, stream=True)
    classes = model.names

    # 結果を保持するための辞書
    goal_info = {}
    goal_info["file_name"] = input_video_name
    goals = []

    x1, y1, x2, y2, x3, y3, x4, y4 = 0,0,0,0,0,0,0,0
    # 各フレームの結果を取得
    for frame_id, result in enumerate(results):
        confidence = 0
        posts = []
        for i, box in enumerate(result.boxes):
            # ゴールのバウンディングボックスを取得（正規化された値で）
            x1, y1, x2, y2 = box.xyxyn[0].tolist()  # リストへ
            confidence = box.conf[0].item()   # 確信度
            posts.append([(x1+x2)/2, y1, (x1+x2)/2, y2, confidence]) # ゴールポストの上と下の座標と確信度をリストに追加

        # ゴールポストが2つ以上検出された場合情報を更新する
        if len(posts) >= 2:
            left_index = np.argmin(np.array([post[0] for post in posts]))
            right_index = np.argmax(np.array([post[0] for post in posts]))
            left_post = posts[left_index]
            right_post = posts[right_index]
            x1, y1, x2, y2, x3, y3, x4, y4 = left_post[0], left_post[1], right_post[0], right_post[1], left_post[2], left_post[3], right_post[2], right_post[3]

        # Create a dictionary for the detected object with frame ID
        goal = {
            "frame_id": frame_id,
            "class_name": "goal",
            "coordinates": {
                "left-up": [x1, y1],
                "right-up": [x2, y2],
                "left-down": [x3, y3],
                "right-down": [x4, y4]
            },
            "confidence": confidence,
        }

        # Add the detected object to the list
        goals.append(goal)

    # ゴールの4角の情報を獲得
    left_up_x = [goal["coordinates"]["left-up"][0] for goal in goals if goal["coordinates"]["left-up"] != [0, 0]]
    left_up_y = [goal["coordinates"]["left-up"][1] for goal in goals if goal["coordinates"]["left-up"] != [0, 0]]
    right_up_x = [goal["coordinates"]["right-up"][0] for goal in goals if goal["coordinates"]["right-up"] != [0, 0]]
    right_up_y = [goal["coordinates"]["right-up"][1] for goal in goals if goal["coordinates"]["right-up"] != [0, 0]]
    left_down_x = [goal["coordinates"]["left-down"][0] for goal in goals if goal["coordinates"]["left-down"] != [0, 0]]
    left_down_y = [goal["coordinates"]["left-down"][1] for goal in goals if goal["coordinates"]["left-down"] != [0, 0]]
    right_down_x = [goal["coordinates"]["right-down"][0] for goal in goals if goal["coordinates"]["right-down"] != [0, 0]]
    right_down_y = [goal["coordinates"]["right-down"][1] for goal in goals if goal["coordinates"]["right-down"] != [0, 0]]

    # 最小の長方形の座標を計算
    x_min = min(left_up_x + left_down_x) if left_up_x or left_down_x else 0
    x_max = max(right_up_x + right_down_x) if right_up_x or right_down_x else 1
    y_min = min(left_up_y + right_up_y) if left_up_y or right_up_y else 0
    y_max = max(left_down_y + right_down_y) if left_down_y or right_down_y else 1

    goal_info["crop_coordinates"] = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max
    }
    goal_info["goal"] = goals
    # Save detected objects to a JSON file
    with open(goal_output_file_path, "w") as json_file:
        json.dump(goal_info, json_file, indent=4)

    print(f"Goal info of {input_video_name} saved to {goal_output_file_path}")
    return goal_output_file_path

# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/making_datasets/goal_segment/goal_segment.py