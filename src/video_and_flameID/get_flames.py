# 動画の全フレームを画像としてdataset/flameに保存

import cv2
import os

# 動画が保存されているフォルダのパス
video_folder = '/home/u01177/video_edit/dataset/video/'
# 出力する画像ファイルの保存先フォルダ
output_folder = '/home/u01177/video_edit/dataset/flame/'

# 動画ファイルのリストを取得
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

# 各動画ファイルに対して処理を行う
for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)

    # 動画を読み込む
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"動画を開けませんでした: {video_file}")
        continue

    # フレームを保存するためのディレクトリを作成
    save_folder = os.path.join(output_folder, video_file[:-4])  # 拡張子を除いた名前
    os.makedirs(save_folder, exist_ok=True)

    frame_id = 0  # フレームIDの初期化
    while True:
        # フレームを取得
        ret, frame = cap.read()
        if not ret:
            print(f"動画のフレーム取得が終了しました: {video_file}")
            break
        
        # フレームを保存（フレームIDをファイル名に含める）
        output_image_path = os.path.join(save_folder, f"{video_file[:-4]}_frame_{frame_id}.jpg")
        cv2.imwrite(output_image_path, frame)

        frame_id += 1  # フレームIDを増やす

    cap.release()
