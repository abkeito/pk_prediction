import json
import torch

def pose_prediction(inputs, outputs, filename):

    predicted_Coordinates = []
    keypoints = ['face', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                      'face-move','left_shoulder-move', 'right_shoulder-move', 'left_elbow-move', 'right_elbow-move', 'left_wrist-move', 'right_wrist-move', 'left_hip-move', 'right_hip-move', 'left_knee-move', 'right_knee-move', 'left_ankle-move', 'right_ankle-move']
    keypoints_size = 26

    for frame_id, input in enumerate(inputs):
        Coordinates = torch.reshape(input, (keypoints_size, 2))
        keeper_pose = dict(zip(keypoints, Coordinates.tolist()))

        frame = {
            "frame_id" : frame_id,
            "data_type" : "input",
            "keeper-pose" : keeper_pose
        }

        predicted_Coordinates.append(frame)

    for frame_id, output in enumerate(outputs):
        Coordinates = torch.reshape(output, (keypoints_size, 2))
        keeper_pose = dict(zip(keypoints, Coordinates.tolist()))

        frame = {
            "frame_id" : frame_id+len(inputs),
            "data_type" : "output",
            "keeper-pose" : keeper_pose
        }

        predicted_Coordinates.append(frame)

    with open(filename, "w") as f:
        json.dump(predicted_Coordinates, f, indent=4)
    
    print(f"Predicted action of the keeper saved to {filename}")
    