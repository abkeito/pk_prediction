# pk_prediction

1. データセットを自動で生成する方法

- ```src/making_datasets/goal_segment/video/video```に動画ファイルを、```src/making_datasets/goal_segment/video/text```に蹴られた瞬間のフレームIDを入れておく。
- ```srun -p p -t 60:00 --gres=gpu:1 --pty poetry run python src/making_datasets/goal_segment/generate_dataset_from_video.py``` を実行する
- ```srun -p p -t 60:00 --gres=gpu:1 --pty poetry run python src/making_datasets/optical_flow/optical_flow.py``` を実行する
-  ```srun -p p -t 60:00 --gres=gpu:1 --pty poetry run python src/making_datasets/optical_flow/generate_new_dataset.py``` を実行する

2. モデルの学習
- ```srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/prediction/model_train.py``` を実行すると、モデルが生成される

3. モデルのテスト
- ```srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/prediction/model_train.py``` を実行する。

4. 可視化して確認
- ```srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/prediction/display_points_and_move.py``` を実行す流。可視化したい時系列jsonファイルをソースに直書きする。勝手に動画ができている。
