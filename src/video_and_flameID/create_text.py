import os

# ボールを蹴る瞬間のフレームIDを格納したファイル'flameID.txt'へのパス
input_file_path = '/home/u01177/video_edit/flameID.txt' 

# 出力先ディレクトリのパス
output_dir = '/home/u01177/video_edit/dataset/text/'

# 出力先ディレクトリが存在しない場合は作成
os.makedirs(output_dir, exist_ok=True)

# テキストファイルから値を読み込む
with open(input_file_path, 'r') as file:
    values = file.readlines()

# 各値を別のファイルに書き出す
for index, value in enumerate(values):
    # 先頭と末尾の空白文字を削除
    cleaned_value = value.strip()
    # 出力ファイルのパスを生成
    output_file_path = os.path.join(output_dir, f"{index + 1}.txt")
    # 値を書き込む
    with open(output_file_path, 'w') as output_file:
        output_file.write(cleaned_value)
