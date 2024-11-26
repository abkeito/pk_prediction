1.youtubeからyt-dlpを用いてPK動画をダウンロードし、src/get_video_and_frame_id/datao/original_video に保存

2.3秒のPKシーンの秒数をextract_seconds.txtに配列として保存(手動)

3.extract_pk.pyで、source_videoの動画をextract_seconds.txtの配列を用いて3秒の動画に切り取り、src/get_video_and_frame_id/data/video に保存

4.get_frames.pyで、dataset/videoの3秒の動画の全フレームをそれぞれフレームIDと共に画像化し、src/get_video_and_frame_id/data/video_frames に保存

5.蹴った瞬間のフレームIDを確かめ、flameID.txtに保存(手動)

6.create_text.pyで、flameID.txtに保存されたフレームIDをdataset/textに保存
