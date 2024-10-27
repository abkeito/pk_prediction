import json
import torch

def pose_prediction(inputs, outputs, filename):

    
    keypoints = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    keypoints_size = 17
    frame_id = 0

    for input, output in zip(inputs, outputs):
        predicted_coodinates = []
        for frame in input:
            coodinates = torch.reshape(frame, (keypoints_size, 2))
            keeper_pose = dict(zip(keypoints, coodinates.tolist()))

            keeper_pose = {
                "frame_id" : frame_id,
                "data_type" : "input",
                "keeper-pose" : keeper_pose
            }

            predicted_coodinates.append(keeper_pose)
            frame_id += 1

        for frame in output:
            coodinates = torch.reshape(frame, (keypoints_size, 2))
            keeper_pose = dict(zip(keypoints, coodinates.tolist()))

            frame = {
                "frame_id" : frame_id,
                "data_type" : "output",
                "keeper-pose" : keeper_pose
            }

            predicted_coodinates.append(frame)
            frame_id += 1

        with open(filename, "w") as f:
            json.dump(predicted_coodinates, f, indent=4)
    
        print(f"Predicted action of the keeper saved to {filename}")
    