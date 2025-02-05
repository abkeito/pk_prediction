import matplotlib.pyplot as plt
import json
from matplotlib.backends.backend_pdf import PdfPages

# 入力ファイルとPDFの出力ファイルを設定
number = 6
input_file = f"src/transformer_prediction/predicted/{number}_prediction.json"  # フレームデータのJSONファイル
# input_file = f"src/transformer_prediction/data/test/{number}_dataset.json"
pdf_output_file = f"data/prediction/{number}_prediction.pdf"  # PDFの出力ファイル
specific_frame_ids = range(100)  # 出力したいフレームのインデックス

# フレームデータを読み込み
with open(input_file, 'r', encoding='utf-8') as file:
    frames_data = json.load(file)

# ゴール枠の定義（スケーリング済み座標）
goal_top_left = (0, 0)           # ゴールの左上
goal_bottom_right = (7.32, 2.44) # ゴールの右下

# スケール因子
scale_x = 100
scale_y = 100
offset_x = 0
offset_y = 0

def scale_coordinates(coord):
    """座標をスケーリングして描画範囲に変換"""
    x = coord[0] * scale_x + offset_x
    y = coord[1] * scale_y + offset_y
    return x, y

# PDFに描画
pdf_pages = PdfPages(pdf_output_file)

for frame_id in specific_frame_ids:
    if frame_id >= len(frames_data):
        print(f"フレームID {frame_id} は範囲外です。スキップします。")
        continue

    frame = frames_data[frame_id]
    poses, moves = list(frame['keeper-pose'].items())[:13], list(frame['keeper-pose'].items())[13:]
    data_type = frame['data_type']

    fig, ax = plt.subplots(figsize=(6, 3))  # サイズを調整してコンパクトに

    # ゴール枠を描画
    scaled_top_left = scale_coordinates(goal_top_left)
    scaled_bottom_right = scale_coordinates(goal_bottom_right)
    ax.add_patch(
        plt.Rectangle(
            scaled_top_left,
            scaled_bottom_right[0] - scaled_top_left[0],
            scaled_bottom_right[1] - scaled_top_left[1],
            fill=False,
            edgecolor="green",
            linewidth=2,
            label="Goal Frame"
        )
    )

    # 関節と矢印を描画
    if frame['keeper-pose']:
        for pose, move in zip(poses, moves):
            pose_part, pose_coords = pose
            move_part, move_coords = move
            start_x, start_y = scale_coordinates(pose_coords)
            end_x, end_y = scale_coordinates(
                (pose_coords[0] + 10 * move_coords[0], pose_coords[1] + 10 * move_coords[1])
            )

            # 関節を描画
            ax.plot(start_x, start_y, "o", color="red" if data_type == "input" else "blue", label="Joint" if pose == poses[0] else "")
            
            # # 矢印を描画
            # ax.arrow(
            #     start_x, start_y, end_x - start_x, end_y - start_y,
            #     head_width=2, head_length=3, fc="blue", ec="blue", label="Optical Flow" if pose == poses[0] else ""
            # )

            # ラベルを矢印の終点付近に配置し、見やすく調整
            ax.text(
                start_x + 10, start_y, pose_part, fontsize=6, color="black", ha="left", va="center"
            )

    # 描画範囲の設定をゴール枠に合わせてコンパクトに
    ax.set_xlim(scaled_top_left[0], scaled_bottom_right[0])
    ax.set_ylim(scaled_bottom_right[1], scaled_top_left[1])
    ax.set_aspect("equal")
    ax.set_title(f"Frame ID: {frame_id}", fontsize=10)
    ax.legend()
    ax.axis("off")

    # PDFに保存
    # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    # 自動的にレイアウト調整を行い、余白を最小限にする
    plt.tight_layout(pad=0.2)  # pad: 余白の調整
    pdf_pages.savefig(fig)
    plt.close(fig)

# PDFを保存
pdf_pages.close()
print(f"指定したフレームが {pdf_output_file} に保存されました。")

# poetry run python src/making_datasets/utils/display_pdf.py