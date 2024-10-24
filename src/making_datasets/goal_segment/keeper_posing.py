import json
import numpy as np
from ultralytics import YOLO
import os

import torch
print(torch.version.cuda)
print(torch.__version__)
device = 'gpu' if torch.cuda.is_available() else 'cpu'
print(device)

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
# Predict with the model
input_path = "/home/u01170/AI_practice/pk_prediction/src/making_datasets/goal_segment/video/standard_cropped.mp4/cropped_output.mp4"  # predict on an image
outputfile = f"/home/u01170/AI_practice/pk_prediction/src/making_datasets/goal_segment/data/pose.json"
# Display model information (optional)
model.info()
# Run inference on a video file
results = model(input_path, save=True, stream=True)

keeper_info = []
for frame_id, result in enumerate(results):
    # Check if masks are available in the result
    pose_coordinates = result.keypoints.xy.tolist()
    keeper_pose = dict(zip(person_keypoints, pose_coordinates))
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