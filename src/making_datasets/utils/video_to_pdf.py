import cv2
from PIL import Image

def video_frame_to_pdf(video_path, frame_number, output_pdf_path):
    # 動画を読み込む
    cap = cv2.VideoCapture(video_path)
    
    # 動画のフレーム数を取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 指定したフレームが動画内にあるか確認
    if frame_number >= total_frames or frame_number < 0:
        raise ValueError(f"フレーム番号が範囲外です。0～{total_frames - 1}の範囲で指定してください。")
    
    # 指定フレームへ移動
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # フレームを読み取る
    ret, frame = cap.read()
    if not ret:
        raise ValueError("フレームの読み取りに失敗しました。")
    
    # OpenCV形式の画像をPillow形式に変換
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    
    # 画像をPDFに保存
    image.save(output_pdf_path, "PDF", resolution=100.0)
    
    print(f"フレーム{frame_number}をPDFとして保存しました: {output_pdf_path}")
    
    # リソースを解放
    cap.release()

# 使用例
video_path = "runs/pose/predict564/cropped_1.avi"  # 動画のパス
video_path = "runs/pose/predict530/cropped_19.avi"
video_path = "runs/pose/predict460/cropped_5.avi"
video_path = "runs/pose/predict471/cropped_6.avi"
video_path = "runs/pose/predict555/cropped_41.avi"
video_path = "runs/pose/predict455/cropped_45.avi"
frame_number = 0  # 抽出したいフレーム番号
output_pdf_path = "frame_0.pdf"  # 保存先のPDFパス
video_frame_to_pdf(video_path, frame_number, output_pdf_path)

# poetry run python src/making_datasets/utils/video_to_pdf.py