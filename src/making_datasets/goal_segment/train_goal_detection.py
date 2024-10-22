from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/home/u01170/AI_practice/final_pj/datasets/100datasets/data.yaml", epochs=120, imgsz=640, workers=2)

# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python train_goal_detection.py