import cv2
import numpy as np
import json
from base import CROPPED_VIDEO_FOLDER, INPUT_SPLIT_FOLDER, POSE_OUTPUT_FOLDER, GOAL_OUTPUT_FOLDER, OPTICAL_FOLDER
import os

def optical_flow():
    filenames = os.listdir(GOAL_OUTPUT_FOLDER)

    for filename in filenames:
        filenumber = filename.split(".")[0]

        goal_json_path = os.path.join(GOAL_OUTPUT_FOLDER, filename)
        output_path = os.path.join(OPTICAL_FOLDER, filenumber + "_optical-flow.json")

        with open(goal_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        goal_data = data["goal"]
        crop_coordinates = data["crop_coordinates"]
        # 座標を辞書から抽出
        x_min = crop_coordinates["x_min"]
        x_max = crop_coordinates["x_max"]
        y_min = crop_coordinates["y_min"]
        y_max = crop_coordinates["y_max"]
        # 動画を読み込む
        cap = cv2.VideoCapture(os.path.join(CROPPED_VIDEO_FOLDER, "cropped_" + filenumber + ".mp4"))

        # 動画のプロパティを取得
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 出力動画の設定
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

        # 最初のフレームを取得
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read the video file.")
            cap.release()
            out.release()
            exit()

        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        # 特徴点の設定
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        p0 = cv2.goodFeaturesToTrack(first_gray, mask=None, **feature_params)

        # オプティカルフローのパラメータ
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # オプティカルフローのデータを保存するリスト
        optical_flow_data = []

        # フレームをループしてオプティカルフローを計算
        frame_id = 0  # フレームIDの初期化
        while True:

            ## ゴールのやつ
            if frame_id < len(goal_data):
                goal_coordinates = goal_data[frame_id]["coordinates"]
                goal_lu = [(goal_coordinates["left-up"][0] - x_min) / (x_max - x_min),
                        (goal_coordinates["left-up"][1] - y_min) / (y_max - y_min)]
                goal_ld = [(goal_coordinates["left-down"][0] - x_min) / (x_max - x_min),
                        (goal_coordinates["left-down"][1] - y_min) / (y_max - y_min)]
                goal_ru = [(goal_coordinates["right-up"][0] - x_min) / (x_max - x_min),
                        (goal_coordinates["right-up"][1] - y_min) / (y_max - y_min)]
                goal_rd = [(goal_coordinates["right-down"][0] - x_min) / (x_max - x_min),
                        (goal_coordinates["right-down"][1] - y_min) / (y_max - y_min)]
                goal_centroid = [np.mean(goal_lu[0] + goal_ld[0] + goal_ru[0] + goal_rd[0]), np.mean(goal_lu[1] + goal_ld[1] + goal_ru[1] + goal_rd[1])]

                xa, ya = goal_ld[0] - goal_lu[0], goal_ld[1] - goal_lu[1]
                xb, yb = goal_ru[0] - goal_lu[0], goal_ru[1] - goal_lu[1]

                if ya == 0:
                    ya = 1e-6
                    print("ya is zero")
                if xb == 0:
                    xb = 1e-6
                    print("xb is zero")

                transformation_matrix = np.array([[(7.32 / xb), 0], [-(2.44*yb/ya/xb), 2.44/ya]])
                ## ゴールのやつ

            ret, frame = cap.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # オプティカルフローを計算
            p1, st, err = cv2.calcOpticalFlowPyrLK(first_gray, frame_gray, p0, None, **lk_params)

            # 有効なポイントを取得
            if p1 is not None and p0 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # 矢印を描画し、座標を保存
                before_coordinates = []  # このフレームの座標を格納するリスト
                after_coordinates = []
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()

                    # 座標を0〜1に正規化
                    subtract_value = np.array([goal_lu[0], goal_lu[1]])
                    before_coordinate = np.array([a / width, b / height]) - subtract_value
                    before_coordinate = np.dot(transformation_matrix, before_coordinate.T)
                    after_coordinate = np.array([c / width, d / height]) - subtract_value
                    after_coordinate = np.dot(transformation_matrix, after_coordinate.T)

                    before_coordinates.append(before_coordinate.tolist())
                    after_coordinates.append(after_coordinate.tolist())

                    # 矢印を描画
                    cv2.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)

            # 矢印を描画したフレームを動画ファイルに書き込む
            out.write(frame)

            # フレームのデータを保存
            optical_flow_data.append({'frame_id': frame_id, 'before': before_coordinates, 'after': after_coordinates})

            # フレームを更新
            first_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

            frame_id += 1  # フレームIDをインクリメント

        # 後処理
        cap.release()
        out.release()  # 動画ファイルを保存

        # JSONファイルにオプティカルフローの座標を保存
        with open(output_path, 'w') as json_file:
            json.dump(optical_flow_data, json_file, indent=4)

        print(f"Optical flow coordinates saved to {output_path}")

# poetry run python src/making_datasets/optical_flow/optical_flow.py