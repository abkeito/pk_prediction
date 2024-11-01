1.youtubeからyt-dlpを用いてPK動画をダウンロードし、source_videoに保存

2.3秒のPKシーンの秒数をextract_seconds.txtに配列として保存(手動)

3.extract_pk.pyで、source_videoの動画をextract_seconds.txtの配列を用いて3秒の動画に切り取り、dataset/videoに保存

4.get_flames.pyで、dataset/videoの3秒の動画の全フレームをそれぞれフレームIDと共に画像化し、dataset/flameに保存

5.蹴った瞬間のフレームIDを確かめ、flameID.txtに保存

6.create_text.pyで、flameID.txtに保存されたフレームIDをdataset/textに保存