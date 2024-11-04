import json
import numpy as np
from ultralytics import YOLO
import os
import torch
from base import CROPPED_VIDEO_FOLDER, INPUT_SPLIT_FOLDER, POSE_OUTPUT_FOLDER, GOAL_OUTPUT_FOLDER

# これをはみ出たらだめ
x_min, x_max = 0-1, 7.32+1
y_min, y_max = 0-1, 2.44+1

# 座標が範囲内に収まっているかを確認
def are_coordinates_within_bounds(coordinates, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max):
    x_coords = coordinates[0, :]  # x座標
    y_coords = coordinates[1, :]  # y座標

    # x座標とy座標が指定した範囲内にあるかを確認
    within_x_bounds = np.all((x_coords >= x_min) & (x_coords <= x_max))
    within_y_bounds = np.all((y_coords >= y_min) & (y_coords <= y_max))

    return within_x_bounds and within_y_bounds

def keeper_posing(input_video_name: str, goal_output_file_path: str, split_frame: int, input_frame: int, output_frame:int):
    print(torch.version.cuda)
    print(torch.__version__)
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    print(device)

    # Predict with the model
    input_path = os.path.join(CROPPED_VIDEO_FOLDER, "cropped_" + input_video_name)
    input_text_path = os.path.join(INPUT_SPLIT_FOLDER, input_video_name.split(".")[0]+".txt")
    outputfile = os.path.join(POSE_OUTPUT_FOLDER, f"{input_video_name}_pose.json")
    goal_json_path = goal_output_file_path

    with open(goal_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    with open(input_text_path, "r") as file:
        content = file.read()
        print(content)
        kick_frame_id = int(content)

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
        "face", # 0~4個目を1つにまとめる
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
    keeper_pose = None
    pose_coordinates_transformed = None
    for frame_id, result in enumerate(results):
        prev_pose_coordinates_transformed = pose_coordinates_transformed
        # goal の情報
        if frame_id < len(goal_data):
            goal_coordinates = goal_data[frame_id]["coordinates"]
            goal_lu = [(goal_coordinates["left-up"][0] - x_min) / (x_max - x_min),
                    (goal_coordinates["left-up"][1] - y_min) / (y_max - y_min)]
            goal_ld = [(goal_coordinates["left-down"][0] - x_min) / (x_max - x_min),
                    (goal_coordinates["left-down"][1] - y_min) / (y_max - y_min)]
            goal_ru = [(goal_coordinates["right-up"][0] - x_min) / (x_max - x_min),
                    (goal_coordinates["right-up"][1] - y_min) / (y_max - y_min)]
            goal_rd = [(goal_coordinates["right-down"][0] - x_min) / (x_max - x_min),
                    (goal_coordinates["right-down"][1] - y_min) / (y_max - y_min)]
            goal_centroid = [np.mean(goal_lu[0] + goal_ld[0] + goal_ru[0] + goal_rd[0]), np.mean(goal_lu[1] + goal_ld[1] + goal_ru[1] + goal_rd[1])]

            xa, ya = goal_ld[0] - goal_lu[0], goal_ld[1] - goal_lu[1] # クロップ後の動画の中での0~1ベクトル
            xb, yb = goal_ru[0] - goal_lu[0], goal_ru[1] - goal_lu[1]

            if ya == 0:
                ya = 1e-6
                print("ya is zero")
            if xb == 0:
                xb = 1e-6
                print("xb is zero")

            transformation_matrix = np.array([[(7.32 / xb), 0], [-(2.44*yb/ya/xb), 2.44/ya]])
            closest_index = 0

            # キーパーのindexをとってくる
            if result.boxes.xywh.shape != (0,4):
                boxes = result.boxes
                box_areas = []
                for box in boxes:
                    _,_, w, h = box.xywh[0].tolist()
                    box_areas.append(w*h)      
                # 最小の距離を持つ点のインデックスを取得
                closest_index = np.argmax(np.array(box_areas))

            pose_coordinates = np.array(result.keypoints.xyn.tolist()[closest_index])

            # 形が0ではないかcheck
            if pose_coordinates.shape != (0,):
                subtract_value = np.array([goal_lu[0], goal_lu[1]])
                pose_coordinates = np.where(np.all(pose_coordinates == [0, 0], axis=1, keepdims=True), 
                                        pose_coordinates, 
                                        pose_coordinates - subtract_value)
                pose_coordinates_transformed = np.dot(transformation_matrix, pose_coordinates.T).T

                not_points_count = 0

                # face を一つにまとめる
                # print(pose_coordinates_transformed[:5])
                face_valid_points = pose_coordinates_transformed[:5][~np.all(pose_coordinates_transformed[:5] == [0, 0], axis=1)]
                valid_points = pose_coordinates_transformed[~np.all(pose_coordinates_transformed == [0, 0], axis=1)]
                body_centroid = np.mean(valid_points, axis=0).reshape(1, -1)
                if face_valid_points.size > 0:
                    face_centroid = np.mean(face_valid_points, axis=0).reshape(1, -1)
                else:
                    face_centroid = body_centroid.reshape(1, -1) # 有効な座標がない場合、重心を[0, 0]とする
                # なかったら前回を引き継ぐ or 全身の重心
                for i in range(len(pose_coordinates_transformed)):
                    if np.array_equal(pose_coordinates_transformed[i], [0, 0]) and prev_pose_coordinates_transformed is not None and not np.array_equal(prev_pose_coordinates_transformed[i], [0, 0]):
                        not_points_count += 1
                        pose_coordinates_transformed[i][0], pose_coordinates_transformed[i][1] = prev_pose_coordinates_transformed[i][0], prev_pose_coordinates_transformed[i][1]
                    elif np.array_equal(pose_coordinates_transformed[i], [0, 0]):
                        pose_coordinates_transformed[i][0], pose_coordinates_transformed[i][1] = body_centroid[0][0], body_centroid[0][1]

                pose_coordinates_13 = np.concatenate((face_centroid, pose_coordinates_transformed[5:]), axis=0)
                if are_coordinates_within_bounds(pose_coordinates_transformed.T) and not_points_count <= 14: 
                    #読み込まれているの少なすぎたら前回の引きつぐ
                    keeper_pose = dict(zip(person_keypoints, pose_coordinates_13.tolist()))

                
            if frame_id >= kick_frame_id - (split_frame + input_frame) and frame_id < kick_frame_id - split_frame:
                data_type = "input"
            elif frame_id >= kick_frame_id - split_frame and frame_id < kick_frame_id - split_frame + output_frame:
                data_type = "output"
            else:
                data_type = None
            keeper = {
                "frame_id" : frame_id,
                "data_type" : data_type,
                "keeper-pose" : keeper_pose
            }
            keeper_info.append(keeper)

    # Save detected objects to a JSON file
    with open(outputfile, "w") as json_file:
        json.dump(keeper_info, json_file, indent=4)

    print(f"Detected objects with masks saved to {outputfile}")

# keeper_posing("102.mp4", os.path.join(GOAL_OUTPUT_FOLDER, "102.mp4_goal.json"), 5, 15, 15)
# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/making_datasets/goal_segment/keeper_posing.py