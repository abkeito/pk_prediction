import json
import torch

def pose_prediction(inputs, outputs, input_files):
    print(input_files)

<<<<<<< HEAD
    predicted_Coordinates = []
    keypoints = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    keypoints_size = 17

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
=======
>>>>>>> main
    
    keypoints = ['face', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'face-move', 'left_shoulder-move', 'right_shoulder-move', 'left_elbow-move', 'right_elbow-move', 'left_wrist-move', 'right_wrist-move', 'left_hip-move', 'right_hip-move', 'left_knee-move', 'right_knee-move', 'left_ankle-move', 'right_ankle-move']
    keypoints_size = len(keypoints)

    for i, (input, output) in enumerate(zip(inputs, outputs)):
        frame_id = 0
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

        with open("src/prediction/data/predict/" + input_files[i][:-12] + "prediction.json", "w") as f:
            json.dump(predicted_coodinates, f, indent=4)
    
        print(f"Predicted action of the keeper saved to src/prediction/data/predict/" + input_files[i][:-12] + "prediction.json")
    