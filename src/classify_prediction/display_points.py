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


# フレームサイズの設定
frame_width = 640
frame_height = 480
RESULT_FOLDER_PATH = "/home/u01170/AI_practice/pk_prediction/src/classify_prediction/test_result"
VIDEO_FOLDER_PATH = "/home/u01170/AI_practice/pk_prediction/src/classify_prediction/result_video"
ORIGINAL_FOLDER = "/home/u01170/AI_practice/pk_prediction/src/classify_prediction/data/original"
filenames = os.listdir(RESULT_FOLDER_PATH)

for filename in filenames:
    # データセットの選択
    output_file = os.path.join(VIDEO_FOLDER_PATH, filename.split("_")[0] + "_result.mp4")
    filepath = os.path.join(RESULT_FOLDER_PATH, filename)
    original_path = os.path.join(ORIGINAL_FOLDER, filename.split("_")[0] + "_dataset.json")

    with open(filepath, 'r', encoding='utf-8') as file:
        result = json.load(file)["result"]

    with open(original_path, 'r', encoding='utf-8') as file:
        original_data = json.load(file)



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
    for frame in original_data:
        img = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255  # 白背景

        # タイトルを描画
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)  # 黒色のタイトル
        # 枠を描画（矩形の線を引く）
        cv2.rectangle(img, scaled_top_left, scaled_bottom_right, color=(0, 255, 0), thickness=2)  # 緑色の枠
        color = get_color(frame["data_type"])

        ## 予測を書く色の濃さで
        if frame["data_type"] == "output":
            # 長方形の数と各長方形の高さ
            height, width = 2.44, 7.32
            YOKO, TATE = 4, 2
            rect_height = height / TATE
            rect_width = width / YOKO

            # 各長方形を描画
            for i, value in enumerate(result[0]):            
                # 色の濃さを計算 (0-255の範囲にマッピング)
                color_intensity = int(256-int(value * 255) *2 / 3)
                color = (color_intensity, color_intensity, 0)
                
                # 長方形の座標
                top_left = (int((rect_width*(i%4))* 80 + 5), int((rect_height*(i//4)) * 80 + 200))
                bottom_right = (int((rect_width*(i%4)+rect_width)* 80 + 5), int((rect_height*(i//4)+rect_height) * 80 + 200))
                
                # 長方形の描画
                cv2.rectangle(img, top_left, bottom_right, color, -1)

        # 各部位の描画
        poses, moves = list(frame['keeper-pose'].items())[:13], list(frame['keeper-pose'].items())[13:]
        if (frame['keeper-pose']):
            ## 予測を描画する色の濃さで描画する
            for pose, move in zip(poses, moves):
                pose_part, pose_coords = pose
                move_part, move_coords = move
                # スケーリングのため、x座標を80倍、y座標を80倍にする
                x = int(pose_coords[0] * 80 + 50)  # スケーリング
                y = int(pose_coords[1] * 80 + 200)  # スケーリング
                end_x = int((pose_coords[0] + 10*move_coords[0]) * 80 + 50)
                end_y = int((pose_coords[1] + 10*move_coords[1]) * 80 + 200)
                if x < frame_width and y < frame_height:  # フレーム内に収まるか確認
                    cv2.circle(img, (x, y), radius=5, color=color, thickness=-1)  # 部位を赤い点で描画
                    # 部位のラベルを描画
                    cv2.putText(img, pose_part, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)  # 黒色のラベル
                    # 矢印を描画
                    cv2.arrowedLine(img, (x, y), (end_x, end_y), color, 1, tipLength=0.2)
                else:
                    print(x, y, "Outside the boundary")
        
        out.write(img)  # フレームを動画に追加

    # 動画ライターを解放
    out.release()

    print(f"動画が {output_file} に保存されました。")

# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/classify_prediction/display_points.py