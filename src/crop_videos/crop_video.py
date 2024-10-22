import os
from moviepy.editor import VideoFileClip, vfx

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
        goal_top_left = tuple(coordinates['左上'])
        goal_top_right = tuple(coordinates['右上'])
        goal_bottom_left = tuple(coordinates['左下'])
        goal_bottom_right = tuple(coordinates['右下'])

        # 4隅の座標からクロップ範囲を計算
        x_min = min(goal_top_left[0], goal_bottom_left[0])
        x_max = max(goal_top_right[0], goal_bottom_right[0])
        y_min = min(goal_top_left[1], goal_top_right[1])
        y_max = max(goal_bottom_left[1], goal_bottom_right[1])

        # MoviePyを使って動画をクロップ
        with VideoFileClip(self.input_video) as video:
            # 指定範囲でクロップ
            cropped_video = video.fx(vfx.crop, x1=x_min, y1=y_min, x2=x_max, y2=y_max)
            
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

coordinates = {
    '左上': [100, 200],
    '右上': [400, 200],
    '左下': [100, 400],
    '右下': [400, 400]
}
crop_video_with_coordinates("/home/u01177/dataset_pk/data_pk_59.mp4", "/home/u01177/crop_test", coordinates)

