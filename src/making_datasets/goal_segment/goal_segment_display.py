import cv2
import json
import numpy as np

# JSONファイルの読み込み
with open("goal_segment_standard.json", "r") as json_file:
    detected_objects = json.load(json_file)

# 動画ファイルの読み込み
cap = cv2.VideoCapture("standard.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)  # フレームレート取得
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 出力用動画ファイルの設定
output_file = "output_video_with_goals_standard.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# フレームごとに処理
for obj in detected_objects:
    frame_id = obj["frame_id"]
    ret, frame = cap.read()
    
    # フレームが存在するか確認
    if not ret:
        break

    class_name = obj["class_name"]
    coordinates = obj["coordinates"]
    
    # 各座標を取得
    left_up = tuple(map(int, coordinates["left-up"]))  # 左上
    right_up = tuple(map(int, coordinates["right-up"]))  # 右上
    left_down = tuple(map(int, coordinates["left-down"]))  # 左下
    right_down = tuple(map(int, coordinates["right-down"]))  # 右下（typo修正）

    # 平行四辺形の頂点を指定
    points = np.array([left_up, right_up, right_down, left_down], dtype=np.int32)

    # 平行四辺形を描画
    cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)  # 緑色の平行四辺形
    cv2.putText(frame, f'{class_name}: {obj["confidence"]:.2f}', 
                (left_up[0], left_up[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 出力動画にフレームを追加
    out.write(frame)

# リソースの解放
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video with goals saved as {output_file}")
