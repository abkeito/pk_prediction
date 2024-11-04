from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os

# 切り抜きたい動画のファイルパス
input_video = "/home/u01177/video_edit/source/Paraguay v Japan： Full Penalty Shoot-out ｜ 2010 #FIFAWorldCup Round of 16 [Mger-g-Swbo].mp4"

# 出力先フォルダ
output_folder = "/home/u01177/video_edit/dataset/video/"  # 出力フォルダを指定

# フォルダが存在しない場合は作成
os.makedirs(output_folder, exist_ok=True)

# "extract_seconds.txt"から、PKの開始時間と終了時間を指定（秒単位）
pk_segments = [(2,5),(53,56),(93,96),(137,140),(183,186),(221,224),(272,275),(308,311),(351,354)]

for i, (start_time, end_time) in enumerate(pk_segments):
    
    # 他の動画とかぶらないようにindexは適宜調整
    output_filename = os.path.join(output_folder, f"{i+1}.mp4") 
     
    ffmpeg_extract_subclip(input_video, start_time, end_time, targetname=output_filename)
    print(f"Extracted: {output_filename}")
