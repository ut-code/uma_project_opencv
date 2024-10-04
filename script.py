import os
import cv2
import torch
import datetime
import numpy as np
from absl import app, flags
from absl.flags import FLAGS
from ultralytics import YOLOv10
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from pytesseract import pytesseract
from matplotlib import pyplot as plt

# Tesseractのパス設定（必要に応じて変更）
pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# モデルとファイルの設定
model_file = "/content/drive/MyDrive/keiba-project-test/weights/yolov10x.pt"
video_file = "/content/drive/MyDrive/keiba-project-test/ped_track.mp4"
output_file = "/content/drive/MyDrive/keiba-project-test/Output.mp4"
conf = 0.5
class_id = 0
blur_id = None

# Initialize the video capture and the video writer objects
video_cap = cv2.VideoCapture(video_file)
frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_cap.get(cv2.CAP_PROP_FPS))

# Initialize the video writer object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Initialize the DeepSort tracker
tracker = DeepSort(max_age=20, n_init=3)

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using device is {device}')

# Load the YOLO model
model = YOLOv10(model_file)

# Load the COCO class labels the YOLO model was trained on
classes_path = "/content/drive/MyDrive/keiba-project-test/coco.names"
with open(classes_path, "r") as f:
    class_names = f.read().strip().split("\n")

# Create a list of random colors to represent each class
np.random.seed(42)  # to get the same colors
colors = np.random.randint(0, 255, size=(len(class_names), 3))  # (80, 3)

# 初期化（各馬の前フレームでの中心座標を保存）
previous_centers = {}

horse_class_id = class_names.index("horse")

delta_x, delta_y = 1, 0 
x_0 = frame_width / 2
y_0 = frame_height * 1.5
r = frame_height * 1.5

# 直線の係数を計算
a = delta_y
b = -delta_x
c = -(a * x_0 + b * y_0)

# 仮想ラインの初期化
previous_center = None

while True:
    start = datetime.datetime.now()
    ret, frame = video_cap.read()

    if not ret:
        print("End of the video file...")
        break

    
    results = model(frame, verbose=False)[0]
    detect = []
    for det in results.boxes:
        confidence = det.conf
        label = det.cls
        bbox = det.xyxy[0]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if class_id is None:
            if confidence < conf:
                continue
        else:
            if class_id != class_id or confidence < conf:
                continue

        if class_id == horse_class_id and confidence >= conf:
            detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    tracks = tracker.update_tracks(detect, frame=frame)
    horse_positions = []

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        class_id = track.get_det_class()

        # クラスIDが「horse」の場合のみ順位を決定
        if class_id == horse_class_id:
            x1, y1, x2, y2 = map(int, ltrb)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            horse_positions.append((track_id, center_x, center_y))

            # 進行方向の計算
            if previous_center is not None:
                prev_x, prev_y = previous_center
                delta_x = center_x - prev_x
                delta_y = center_y - prev_y
            else:
                delta_x, delta_y = 0, 0  # 初回フレームでは変更なし

            previous_center = (center_x, center_y)  # 現在の中心を保存

            if track_id in previous_centers:
                previous_centers[track_id] = (center_x, center_y)

    # 進行方向の直線の係数を計算
    if previous_center is not None:
        a = delta_y
        b = -delta_x
        c = -(a * x_0 + b * y_0 + math.sqrt((a * x_0 + b * y_0)**2 - (a**2 + b**2) * (-r**2)))

        # 仮想ラインまでの距離を計算
        def calculate_distance_to_line(center_x, center_y):
            return abs(a * center_x + b * center_y + c) / math.sqrt(a**2 + b**2)

        # 仮想ラインまでの距離が近い順に並べ替え
        horse_positions.sort(key=lambda hp: calculate_distance_to_line(hp[1], hp[2]))

        # 仮想ラインを描画
        line_start = (0, int(-(a * 0 + c) ))  # 画面の左端でのy座標
        line_end = (frame_width, int(-(a * frame_width + c) ))  # 画面の右端でのy座標
        cv2.line(frame, line_start, line_end, (0, 255, 0), 2)

        # 順位を描画
        for i, horse_position in enumerate(horse_positions):
            if len(horse_position) == 4:
                track_id, center_x, center_y, _ = horse_position
            elif len(horse_position) == 3:  # もし要素が3つしかない場合の処理
                track_id, center_x, center_y = horse_position
                _ = None  # 移動角度がない場合は None とする
            else:
                print(f"Unexpected number of elements in horse_positions[{i}]: {len(horse_position)}")
                continue
            rank = i + 1
            text = f"Top{rank} ID{track_id}"

            for track in tracks:
                if track.track_id == track_id:
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)
                    color = colors[class_id]
                    B, G, R = map(int, color)

                    # 順位を描画
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                    #cv2.rectangle(frame, (x1 - 1, y1 - 40), (x1 + len(text) * 12, y1), (B, G, R), -1)
                    #cv2.putText(frame, text, (x1 + 5, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 一位の馬を表示する部分
        """
        if len(horse_positions) > 0:
            top_ranked_horse = horse_positions[0]
            top_track_id = top_ranked_horse[0]
            if len(top_ranked_horse) >= 3:
                top_center_x, top_center_y = top_ranked_horse[1:3]
                top_text = f"1st-ID: {top_track_id}"
                cv2.putText(frame, top_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        """

    # 正方形内の数字の読み取り
    """
    x_min_square, x_max_square = 300, 500
    y_min_square, y_max_square = 800, 1000
    square_roi = frame[y_min_square:y_max_square, x_min_square:x_max_square]
    square_text = pytesseract.image_to_string(square_roi, config='--psm 6 digits')
    lines = square_text.split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        if line:
            print(f"正方形内の数字: {line}")
            cv2.putText(frame, line, (x_min_square, y_min_square + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    """

    end = datetime.datetime.now()
    print(f"Time to process 1 frame: {(end - start).total_seconds():.2f} seconds")
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
    writer.write(frame)

# 動画キャプチャとライターの解放
video_cap.release()
writer.release()
  

