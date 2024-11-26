from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os

# 切り抜きたい動画のファイルパス
input_video = "src/get_video_and_frame_id/data/original_video/47.mp4"

# 出力先フォルダ
output_folder = "src/get_video_and_frame_id/data/video"  # 出力フォルダを指定

# フォルダが存在しない場合は作成
os.makedirs(output_folder, exist_ok=True)

# "extract_seconds.txt"から、PKの開始時間と終了時間を指定（秒単位）
pk_segments = [(1,3)]

for i, (start_time, end_time) in enumerate(pk_segments):
    
    # 他の動画とかぶらないようにindexは適宜調整
    output_filename = os.path.join(output_folder, f"{i+1}.mp4") 
     
    ffmpeg_extract_subclip(input_video, start_time, end_time, targetname=output_filename)
    print(f"Extracted: {output_filename}")

# poetry run python src/get_video_and_frame_id/extract_pk.py