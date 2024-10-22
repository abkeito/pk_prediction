import json
import numpy as np
from ultralytics import SAM, YOLO

import numpy as np

# Load a model
#model = YOLO("yolo11x-pose.pt")
model = YOLO("/home/u01170/AI_practice/final_pj/project/goal_detection/runs/detect/train13/weights/best.pt")
import torch
print(torch.version.cuda)
print(torch.__version__)
device = 'gpu' if torch.cuda.is_available() else 'cpu'
print(device)
# Load a model
# model = SAM("sam_b.pt")

# Display model information (optional)
model.info()
# Run inference on a video file
results = model("left2.png", save=True, stream=True)
classes = model.names

# Prepare a list to hold detected objects
detected_objects = []
masks = []
# Extract the bounding boxes, masks, and segmentations
for frame_id, result in enumerate(results):
    # Check if masks are available in the result
    goals = {}
    
    for i, box in enumerate(result.boxes):
        # Get the bounding box coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert to list if needed
        confidence = box.conf[0].item()  # Confidence score
        class_id = box.cls[0].item()  # Class ID
        class_name = classes[class_id]
        goals[class_name] = [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]

    if "soccer-goal" in goals.keys() and  "inner-soccer-goal" in goals.keys():
        if goals["inner-soccer-goal"][0] < goals["soccer-goal"][0]: #左上のパターン
            x1, y1 = goals["soccer-goal"][0] - (goals["soccer-goal"][2] / 2) , goals["soccer-goal"][1] - (goals["soccer-goal"][3] / 2) 
            x2, y2 = x1 + (goals["inner-soccer-goal"][2] * 2) , y1 + (goals["inner-soccer-goal"][3] * 2)
            x4, y4 = goals["soccer-goal"][0] + (goals["soccer-goal"][2] / 2) , goals["soccer-goal"][1] + (goals["soccer-goal"][3] / 2) 
            x3, y3 = x4 - (goals["inner-soccer-goal"][2] * 2) , y4 - (goals["inner-soccer-goal"][3] * 2)
        else:
            x2, y2 = goals["soccer-goal"][0] + (goals["soccer-goal"][2] / 2) , goals["soccer-goal"][1] - (goals["soccer-goal"][3] / 2) 
            x1, y1 = x2 - (goals["inner-soccer-goal"][2] * 2) , y2 + (goals["inner-soccer-goal"][3] * 2)
            x3, y3 = goals["soccer-goal"][0] - (goals["soccer-goal"][2] / 2) , goals["soccer-goal"][1] + (goals["soccer-goal"][3] / 2) 
            x4, y4 = x3 + (goals["inner-soccer-goal"][2] * 2) , y3 - (goals["inner-soccer-goal"][3] * 2)

        # Create a dictionary for the detected object with frame ID
        detected_object = {
            "frame_id": frame_id,
            "class_name": classes[0],
            "coordinates": {
                "left-up": [x1, y1],
                "right-up": [x2, y2],
                "left-down": [x3, y3],
                "right-down": [x4, y4]
            },
            "confidence": confidence,
        }
        # Add the detected object to the list
        detected_objects.append(detected_object)

# Save detected objects to a JSON file
with open("goal_segment.json", "w") as json_file:
    json.dump(detected_objects, json_file, indent=4)

print("Detected objects with masks saved to goal_segment.json")