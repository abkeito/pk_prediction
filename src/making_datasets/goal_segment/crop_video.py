import os
from moviepy.editor import VideoFileClip, vfx
import json

class GoalCropper:
    def __init__(self, input_video, output_folder):
        """
        input_video: 動画ファイルのパス
        output_folder: 出力先のフォルダ
        """
        self.input_video = input_video
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)  # 出力フォルダがなければ作成

    def crop_goal_area(self, coordinates):
        """
        coordinates: {'左上': [x1, y1], '右上': [x2, y2], '左下': [x3, y3], '右下': [x4, y4]}
        辞書形式で座標を受け取り、その範囲で動画をクロップして保存する。
        """
        # 座標を辞書から抽出
        goal_top_left = tuple(coordinates['left-up'])
        goal_top_right = tuple(coordinates['right-up'])
        goal_bottom_left = tuple(coordinates['left-down'])
        goal_bottom_right = tuple(coordinates['right-down'])

        # 4隅の座標からクロップ範囲を計算
        x_min = min(goal_top_left[0], goal_bottom_left[0])
        x_max = max(goal_top_right[0], goal_bottom_right[0])
        y_min = min(goal_top_left[1], goal_top_right[1])
        y_max = max(goal_bottom_left[1], goal_bottom_right[1])

        # MoviePyを使って動画をクロップ
        with VideoFileClip(self.input_video) as video:
            # フレーム数を取得
            total_frames = int(video.fps * video.duration)
            print(f"Total frames in the original video: {total_frames}")
            # 指定範囲でクロップ
            cropped_video = video.fx(vfx.crop, x1=x_min*video.w, y1=y_min*video.h, x2=x_max*video.w, y2=y_max*video.h)
            
            # 出力ファイルの名前を設定
            output_filename = os.path.join(self.output_folder, "cropped_output.mp4")
            
            # クロップした動画を保存
            cropped_video.write_videofile(output_filename, codec='libx264', audio_codec='aac')

        
        print(f"Cropped video saved to: {output_filename}")

# 関数として実装
def crop_video_with_coordinates(input_video_path, output_folder_path, coordinates):
    """
    input_video_path: 動画ファイルのパス
    output_folder_path: 出力先のフォルダパス
    coordinates: ゴールの座標を辞書形式で受け取る。例:
    {'左上': [x1, y1], '右上': [x2, y2], '左下': [x3, y3], '右下': [x4, y4]}
    """
    # GoalCropperクラスのインスタンスを作成
    cropper = GoalCropper(input_video=input_video_path, output_folder=output_folder_path)
    
    # ゴール範囲の動画をクロップ
    cropper.crop_goal_area(coordinates)

# JSONファイルを開く
jsonfile = f"/home/u01170/AI_practice/pk_prediction/src/making_datasets/goal_segment/data/distorted.mp4.json"
with open(jsonfile, 'r', encoding='utf-8') as file:
    data = json.load(file)
coordinates = data["crop_coordinates"]

crop_video_with_coordinates("/home/u01170/AI_practice/pk_prediction/src/making_datasets/goal_segment/video/distorted.mp4",
                            "/home/u01170/AI_practice/pk_prediction/src/making_datasets/goal_segment/video/",
                            coordinates)

# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/making_datasets/goal_segment/crop_video.py