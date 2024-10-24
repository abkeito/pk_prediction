import numpy as np
import matplotlib.pyplot as plt
import cv2
import json

# フレームサイズの設定
frame_width = 640
frame_height = 480
input_file = f"/home/u01170/AI_practice/pk_prediction/src/making_datasets/goal_segment/data/pose.json"
output_file = '/home/u01170/AI_practice/pk_prediction/src/making_datasets/goal_segment/video/display_points.mp4'

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

    # 各部位の描画
    for part, coords in frame['keeper-pose'].items():
        # スケーリングのため、x座標を80倍、y座標を80倍にする
        x = int(coords[0] * 80 + 50)  # スケーリング
        y = int(coords[1] * 80 + 200)  # スケーリング
        if x < frame_width and y < frame_height:  # フレーム内に収まるか確認
            cv2.circle(img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # 部位を赤い点で描画
            # 部位のラベルを描画
            cv2.putText(img, part, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)  # 黒色のラベル
    
    out.write(img)  # フレームを動画に追加

# 動画ライターを解放
out.release()

print(f"動画が {output_file} に保存されました。")

# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/making_datasets/goal_segment/display_points.py