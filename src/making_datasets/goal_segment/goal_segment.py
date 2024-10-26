import json
import numpy as np
from ultralytics import SAM, YOLO
import os
import numpy as np
import torch
from base import GOAL_SEGMENT_MODEL, INPUT_VIDEO_FOLDER, GOAL_OUTPUT_FOLDER

# ビデオを受け取ってゴールの形をjson形式で持ってくる関数
# 入力：ビデオファイルの名前
# 出力：GOAL_OUTPUT_FOLDER 内に json形式が作られる
def goal_segment(input_video_name: str):
    # Load a model
    model = YOLO(GOAL_SEGMENT_MODEL)
    input_path = os.path.join(INPUT_VIDEO_FOLDER, input_video_name)
    goal_output_file_path = os.path.join(GOAL_OUTPUT_FOLDER, input_video_name + "_goal.json")

    print(torch.version.cuda)
    print(torch.__version__)
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    print(device)

    # Display model information (optional)
    model.info()
    # Run inference on a video file
    results = model(input_path, save=True, stream=True)
    classes = model.names

    # Prepare a list to hold detected objects
    goal_info = {}
    goal_info["file_name"] = input_video_name

    goals = []

    goal_frames = 0
    x1, y1, x2, y2, x3, y3, x4, y4 = 0,0,0,0,0,0,0,0
    # Extract the bounding boxes, masks, and segmentations
    for frame_id, result in enumerate(results):
        confidence = 0
        # Check if masks are available in the result
        posts = []
        for i, box in enumerate(result.boxes):
            # Get the bounding box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxyn[0].tolist()  # Convert to list if needed
            confidence = box.conf[0].item()   # Confidence score
            class_id = box.cls[0].item()  # Class ID
            class_name = classes[class_id]
            posts.append([(x1+x2)/2, y1, (x1+x2)/2, y2, confidence]) # postの上と下とconfidenceを入れる
           

        if len(posts) == 2 and posts[0][4] > 0.5 and posts[1][4] > 0.5:
            if posts[0][0] < posts[1][0]:
                left_post, right_post = posts[0], posts[1]
            else:
                left_post, right_post = posts[1], posts[0]
            goal_frames += 1
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
    print(f"frame number = {frame_id}")

    # 座標を辞書から抽出
    goal_top_left = [np.sum([goal["coordinates"]["left-up"][0] for goal in goals]) / goal_frames, np.sum([goal["coordinates"]["left-up"][1] for goal in goals]) / goal_frames]
    goal_top_right = [np.sum([goal["coordinates"]["right-up"][0] for goal in goals])/ goal_frames, np.sum([goal["coordinates"]["right-up"][1] for goal in goals])/ goal_frames]
    goal_bottom_left = [np.sum([goal["coordinates"]["left-down"][0] for goal in goals])/ goal_frames, np.sum([goal["coordinates"]["left-down"][1] for goal in goals])/ goal_frames]
    goal_bottom_right = [np.sum([goal["coordinates"]["right-down"][0] for goal in goals])/ goal_frames, np.sum([goal["coordinates"]["right-down"][1] for goal in goals])/ goal_frames]

    # 4隅の座標からクロップ範囲を計算
    x_range = max(goal_top_right[0], goal_bottom_right[0]) - min(goal_top_left[0], goal_bottom_left[0])
    y_range = max(goal_bottom_left[1], goal_bottom_right[1]) - min(goal_top_left[1], goal_top_right[1])

    x_min = (min(goal_top_left[0], goal_bottom_left[0]) - x_range*0.05) if (min(goal_top_left[0], goal_bottom_left[0]) - x_range*0.05) > 0 else 0
    x_max = (max(goal_top_right[0], goal_bottom_right[0]) + x_range*0.05) if (max(goal_top_right[0], goal_bottom_right[0]) + x_range*0.05) < 1 else 1
    y_min = (min(goal_top_left[1], goal_top_right[1]) - y_range*0.05) if (min(goal_top_left[1], goal_top_right[1]) - y_range*0.05) > 0 else 0
    y_max = (max(goal_bottom_left[1], goal_bottom_right[1]) + y_range*0.05) if (max(goal_bottom_left[1], goal_bottom_right[1]) + y_range*0.05) < 1 else 1

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