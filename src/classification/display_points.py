import json
import numpy as np
import matplotlib.pyplot as plt

def display_points(file_path : str):
    # データの読み込み
    with open(file_path, "r") as file:
        data = json.load(file)
    label = data["label"]

    TATE = int(np.sqrt(len(label) // 3))
    YOKO = TATE*3
    # データを9×3にリシェイプ
    data = np.array(label).reshape((TATE, YOKO))

    # 可視化
    plt.figure(figsize=(YOKO, TATE))
    plt.imshow(data, cmap="Greys", aspect="auto")  # 黒白で可視化
    plt.colorbar(label="Value")  # カラーバー追加
    plt.xticks(range(data.shape[1]), labels=[f"Col {i+1}" for i in range(data.shape[1])])  # X軸のラベル
    plt.yticks(range(data.shape[0]), labels=[f"Row {i+1}" for i in range(data.shape[0])])  # Y軸のラベル
    plt.title("Matrix Visualization")
    plt.show()
    # ファイルに保存
    plt.savefig("data_visualization.png")

    print(data)

if __name__ == "__main__":
    display_points("src/classification/data/test/6_dataset_dataset.json")

# poetry run python src/classification/display_points.py