import os
import shutil
import random

def split_files(input_dir: str, output_dir: str, split_ratios: tuple=(0.7, 0.2, 0.1)) -> None:
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    random.shuffle(files)

    num_total_files = len(files)
    num_train_files = int(num_total_files * split_ratios[0])
    num_val_files = int(num_total_files * split_ratios[1])

    train_files = files[:num_train_files]
    val_files = files[num_train_files:num_train_files + num_val_files]
    test_files = files[num_train_files + num_val_files:]

    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for file in train_files:
        shutil.copy(os.path.join(input_dir, file), os.path.join(train_dir, file))
    for file in val_files:
        shutil.copy(os.path.join(input_dir, file), os.path.join(val_dir, file))
    for file in test_files:
        shutil.copy(os.path.join(input_dir, file), os.path.join(test_dir, file))

    print(f"devided files:")
    print(f"  training set: {len(train_files)}")
    print(f"  validation set: {len(val_files)}")
    print(f"  test set: {len(test_files)}")

input_dir = "src/transformer_prediction/data/all"
output_dir = "src/transformer_prediction/data"
split_files(input_dir, output_dir)

# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/transformer_prediction/devide_files.py 