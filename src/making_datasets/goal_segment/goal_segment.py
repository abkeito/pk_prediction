import json
import numpy as np
from ultralytics import SAM, YOLO
import os
import numpy as np

# Load a model
model = YOLO("/home/u01170/AI_practice/pk_prediction/src/making_datasets/goal_segment/best.pt")
input_video = "distorted.mp4"
input_path = os.path.join("/home/u01170/AI_practice/pk_prediction/src/making_datasets/goal_segment/video/", input_video)
outputfile = f"/home/u01170/AI_practice/pk_prediction/src/making_datasets/goal_segment/data/{input_video}.json"

import torch
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
goal_info["file_name"] = input_video

goals = []

x1, y1, x2, y2, x3, y3, x4, y4 = 0,0,0,0,0,0,0,0
# Extract the bounding boxes, masks, and segmentations
for frame_id, result in enumerate(results):
    # Check if masks are available in the result
    posts = []
    for i, box in enumerate(result.boxes):
        # Get the bounding box coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = box.xyxyn[0].tolist()  # Convert to list if needed
        confidence = box.conf[0].item()  # Confidence score
        class_id = box.cls[0].item()  # Class ID
        class_name = classes[class_id]
        posts.append([(x1+x2)/2, y1, (x1+x2)/2, y2, confidence]) # postの上と下とconfidenceを入れる
    
    if len(posts) == 2 and posts[0][4] > 0.5 and posts[1][4] > 0.5:
        if posts[0][0] < posts[1][0]:
            left_post, right_post = posts[0], posts[1]
        else:
            left_post, right_post = posts[1], posts[0]
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
goal_info["crop_coordinates"] = {
        "left-up": [np.mean([goal["coordinates"]["left-up"][0] for goal in goals]), np.mean([goal["coordinates"]["left-up"][1] for goal in goals])],
        "right-up": [np.mean([goal["coordinates"]["right-up"][0] for goal in goals]), np.mean([goal["coordinates"]["right-up"][1] for goal in goals])],
        "left-down": [np.mean([goal["coordinates"]["left-down"][0] for goal in goals]), np.mean([goal["coordinates"]["left-down"][1] for goal in goals])],
        "right-down": [np.mean([goal["coordinates"]["right-down"][0] for goal in goals]), np.mean([goal["coordinates"]["right-down"][1] for goal in goals])],
}
goal_info["goal"] = goals
# Save detected objects to a JSON file
with open(outputfile, "w") as json_file:
    json.dump(goal_info, json_file, indent=4)

print(f"Detected objects with masks saved to {outputfile}")

# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/making_datasets/goal_segment/goal_segment.py