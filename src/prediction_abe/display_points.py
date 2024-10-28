import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import os

def get_color(data_type):
    if data_type == "input":
        return (0, 255, 0)  # 緑
    elif data_type == "output":
        return (255, 0, 0)  # 赤
    else:
        return (0, 0, 0)    # 黒

# scalerなどを読みとる
with open(os.path.join("/home/u01170/AI_practice/pk_prediction/src/prediction_abe/data_info", "data_and_scaler_info.json"), "r") as f:
    data_info = json.load(f)

test_indices = data_info["Dataset_Indices"]["test_indices"]

# フレームサイズの設定
frame_width = 640
frame_height = 480
input_folder = "/home/u01170/AI_practice/pk_prediction/src/prediction_abe/prediction_data"
output_folder = "/home/u01170/AI_practice/pk_prediction/src/prediction_abe/prediction_video"
#inputfiles = os.listdir("/home/u01170/AI_practice/pk_prediction/src/making_datasets/goal_segment/dataset/dataset_example")

for index in test_indices:
    input_file = os.path.join(input_folder, str(index) + "predict_pose.json")
    output_file = os.path.join(output_folder, str(index) + ".mp4")
    with open(input_file, 'r', encoding='utf-8') as file:
        frames_data = json.load(file)

    # 動画ライターを設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4形式で保存
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (frame_width, frame_height))
    title = "Tracking Goal Keeper in Penalty Kick"

    # 枠の座標（スケーリングのために50倍）
    top_left = (0, 0)           # (0, 0)
    bottom_right = (7.32, 2.44) # (7.32, 2.44)
    scaled_top_left = (int(top_left[0] * 80 + 50), int(top_left[1] * 80 + 200))
    scaled_bottom_right = (int(bottom_right[0] * 80 + 50), int(bottom_right[1] * 80 + 200))

    # 座標を描画して動画を生成
    for frame in frames_data:
        img = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255  # 白背景

        # タイトルを描画
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)  # 黒色のタイトル
        # 枠を描画（矩形の線を引く）
        cv2.rectangle(img, scaled_top_left, scaled_bottom_right, color=(0, 255, 0), thickness=2)  # 緑色の枠
        color = get_color(frame["data_type"])
        # 各部位の描画
        if (frame['keeper-pose']):
            for part, coords in frame['keeper-pose'].items():
                # スケーリングのため、x座標を80倍、y座標を80倍にする
                x = int(coords[0] * 80 + 50)  # スケーリング
                y = int(coords[1] * 80 + 200)  # スケーリング
                if x < frame_width and y < frame_height:  # フレーム内に収まるか確認
                    cv2.circle(img, (x, y), radius=5, color=color, thickness=-1)  # 部位を赤い点で描画
                    # 部位のラベルを描画
                    cv2.putText(img, part, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)  # 黒色のラベル
                else:
                    print(x, y, "Outside the boundary")
        
        out.write(img)  # フレームを動画に追加

    # 動画ライターを解放
    out.release()

    print(f"動画が {output_file} に保存されました。")

# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/prediction_abe/display_points.py