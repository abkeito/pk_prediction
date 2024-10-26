import os
from moviepy.editor import VideoFileClip, vfx
import json
import os
from base import CROPPED_VIDEO_FOLDER, INPUT_VIDEO_FOLDER

class GoalCropper:
    def __init__(self, input_video, output_folder):
        """
        input_video: 動画ファイルのパス
        output_folder: 出力先のフォルダ
        """
        self.input_video = input_video
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)  # 出力フォルダがなければ作成

    def crop_goal_area(self, coordinates, output_filename):
        """
        coordinates: {'左上': [x1, y1], '右上': [x2, y2], '左下': [x3, y3], '右下': [x4, y4]}
        辞書形式で座標を受け取り、その範囲で動画をクロップして保存する。
        """

        # 4隅の座標からクロップ範囲を計算
        x_min = coordinates["x_min"]
        x_max = coordinates["x_max"]
        y_min = coordinates["y_min"]
        y_max = coordinates["y_max"]

        # MoviePyを使って動画をクロップ
        with VideoFileClip(self.input_video) as video:
            # フレーム数を取得
            total_frames = int(video.fps * video.duration)
            print(f"Total frames in the original video: {total_frames}")
            # 指定範囲でクロップ
            cropped_video = video.fx(vfx.crop, x1=x_min*video.w, y1=y_min*video.h, x2=x_max*video.w, y2=y_max*video.h)
            
            # 出力ファイルの名前を設定
            output_file_path = os.path.join(self.output_folder, output_filename)
            
            # クロップした動画を保存
            cropped_video.write_videofile(
                output_file_path,
                codec='libx264',
                audio_codec='aac',
                bitrate="5000k",          # ビットレートを指定
                preset="slow",            # 高画質用のエンコードプリセット
            )
        
        print(f"Cropped video saved to: {output_filename}")

# 関数として実装
def crop_video_with_coordinates(input_video_name, goal_output_file_path):
    """
    input_video_path: 動画ファイルのパス
    output_folder_path: 出力先のフォルダパス
    """
    # GoalCropperクラスのインスタンスを作成
    input_video_path = os.path.join(INPUT_VIDEO_FOLDER, input_video_name)
    output_file_name = "cropped_" + input_video_name
    cropper = GoalCropper(input_video=input_video_path, output_folder=CROPPED_VIDEO_FOLDER)

    # ゴール範囲の動画をクロップ
    with open(goal_output_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    coordinates = data["crop_coordinates"]

    cropper.crop_goal_area(coordinates, output_file_name)


# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/making_datasets/goal_segment/crop_video.py