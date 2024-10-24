import json
import numpy as np
from ultralytics import YOLO
import os
import torch
from base import CROPPED_VIDEO_FOLDER, POSE_OUTPUT_FOLDER

def keeper_posing(input_video_name: str, goal_output_file_path: str):
    print(torch.version.cuda)
    print(torch.__version__)
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    print(device)

    # Predict with the model
    input_path = os.path.join(CROPPED_VIDEO_FOLDER, input_video_name)
    outputfile = os.path.join(POSE_OUTPUT_FOLDER, f"{input_video_name}_pose.json")
    goal_json_path = goal_output_file_path

    with open(goal_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    goal_data = data["goal"]
    crop_coordinates = data["crop_coordinates"]
    # 座標を辞書から抽出
    x_min = crop_coordinates["x_min"]
    x_max = crop_coordinates["x_max"]
    y_min = crop_coordinates["y_min"]
    y_max = crop_coordinates["y_max"]

    # Load a model
    model = YOLO("yolo11x-pose.pt") 
    person_keypoints = [
        "nose",          # 0
        "left_eye",      # 1
        "right_eye",     # 2
        "left_ear",      # 3
        "right_ear",     # 4
        "left_shoulder", # 5
        "right_shoulder",# 6
        "left_elbow",    # 7
        "right_elbow",   # 8
        "left_wrist",    # 9
        "right_wrist",   # 10
        "left_hip",      # 11
        "right_hip",     # 12
        "left_knee",     # 13
        "right_knee",    # 14
        "left_ankle",    # 15
        "right_ankle"    # 16
    ]

    # Display model information (optional)
    model.info()
    # Run inference on a video file
    results = model(input_path, save=True, stream=True)

    keeper_info = []
    for frame_id, result in enumerate(results):
        # goal の情報
        if frame_id < len(goal_data):
            goal_coordinates = goal_data[frame_id]["coordinates"]
            goal_lu = [(goal_coordinates["left-up"][0] - x_min) / (x_max - x_min),
                    (goal_coordinates["left-up"][1] - y_min) / (y_max - y_min)]
            goal_ld = [(goal_coordinates["left-down"][0] - x_min) / (x_max - x_min),
                    (goal_coordinates["left-down"][1] - y_min) / (y_max - y_min)]
            goal_ru = [(goal_coordinates["right-up"][0] - x_min) / (x_max - x_min),
                    (goal_coordinates["right-up"][1] - y_min) / (y_max - y_min)]
            xa, ya = goal_ld[0] - goal_lu[0], goal_ld[1] - goal_lu[1]
            xb, yb = goal_ru[0] - goal_lu[0], goal_ru[1] - goal_lu[1]

            transformation_matrix = np.array([[(7.32 / xb), 0], [-(2.44*yb/ya/xb), 2.44/ya]])

            # 姿勢の座標を変換して登録していく
            pose_coordinates = np.array(result.keypoints.xyn.tolist()[0])
            if pose_coordinates.shape != (0,):
                subtract_value = np.array([goal_lu[0], goal_lu[1]])
                pose_coordinates = np.where(np.all(pose_coordinates == [0, 0], axis=1, keepdims=True), 
                                        pose_coordinates, 
                                        pose_coordinates - subtract_value)
                pose_coordinates_transformed = np.dot(transformation_matrix, pose_coordinates.T)
                keeper_pose = dict(zip(person_keypoints, pose_coordinates_transformed.T.tolist()))
            keeper = {
                "frame_id" : frame_id,
                "keeper-pose" : keeper_pose
            }
            keeper_info.append(keeper)

    # Save detected objects to a JSON file
    with open(outputfile, "w") as json_file:
        json.dump(keeper_info, json_file, indent=4)

    print(f"Detected objects with masks saved to {outputfile}")

# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/making_datasets/goal_segment/keeper_posing.py