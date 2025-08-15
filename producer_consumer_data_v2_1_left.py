import cv2
import mediapipe as mp
import numpy as np
import os
import threading
import queue
import time
from pynput import keyboard

# 설정값
COLLECTION_DURATION = 15
OUTPUT_PATH = './gesture_data_v3_left/'
BASE_FILENAME = 'hybrid_typing_data_v3'
LABEL_DURATION = 0.2  # 키보드 입력 이후 label=1 유지 시간 (초)

# 큐 및 플래그
frame_q = queue.Queue(maxsize=10)
result_q = queue.Queue()
display_q = queue.Queue(maxsize=10)
stop_flag = threading.Event()

# 라벨링 상태 관리 변수
labeling_active = False
last_keypress_time = 0
labeling_lock = threading.Lock()

def on_press(key):
    global labeling_active, last_keypress_time
    with labeling_lock:
        labeling_active = True
        last_keypress_time = time.time()

def get_next_filename(path, basename):
    os.makedirs(path, exist_ok=True)
    index = 3001
    while True:
        filename = os.path.join(path, f"{basename}_{index:04d}.csv")
        if not os.path.exists(filename):
            return filename
        index += 1

def producer():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            frame_q.put(frame, timeout=0.05)
        except queue.Full:
            pass
    cap.release()
    print("[Producer] 종료")

def consumer():
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while not stop_flag.is_set():
            try:
                frame = frame_q.get(timeout=0.05)
            except queue.Empty:
                continue

            image = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks and results.multi_handedness:
                if results.multi_handedness[0].classification[0].label == 'Left':
                    hand_landmarks = results.multi_hand_landmarks[0]
                    try:
                        world_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                        wrist_coords = world_landmarks[0]
                        relative_landmarks = world_landmarks - wrist_coords
                        scale_distance = np.linalg.norm(relative_landmarks[9])
                        if scale_distance > 1e-6:
                            normalized_landmarks = relative_landmarks / scale_distance
                            feature_vector = normalized_landmarks.flatten()
                            result_q.put(feature_vector)
                    except Exception as e:
                        print(f"[Consumer] 오류: {e}")
                    mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            display_q.put(image)
    print("[Consumer] 종료")

def main():
    global labeling_active, last_keypress_time

    print(f"[Main] {COLLECTION_DURATION}초 동안 수집 시작")
    filename = get_next_filename(OUTPUT_PATH, BASE_FILENAME)
    print(f"[Main] 저장 파일: {os.path.basename(filename)}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)
    producer_thread.start()
    consumer_thread.start()

    collected = []
    start_time = time.time()

    while time.time() - start_time < COLLECTION_DURATION:
        current_time = time.time()

        # 라벨 판단
        with labeling_lock:
            if labeling_active and (current_time - last_keypress_time < LABEL_DURATION):
                label = 1
            else:
                label = 0
                labeling_active = False

        try:
            feature = result_q.get_nowait()
            data_row = np.concatenate([feature, [label]])
            collected.append(data_row)
        except queue.Empty:
            pass

        try:
            img = display_q.get_nowait()
            remain = COLLECTION_DURATION - (time.time() - start_time)
            status = "TYPING" if label else "NOT TYPING"
            color = (0, 0, 255) if label else (0, 255, 0)
            cv2.putText(img, f"Time: {remain:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            cv2.putText(img, f"Status: {status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(img, f"Frames: {len(collected)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            try:
                cv2.imshow("Data Collection", img)
            except cv2.error as e:
                print(f"[OpenCV] imshow 에러: {e}")
        except queue.Empty:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("[Main] 수집 종료. 정리 중...")
    stop_flag.set()
    producer_thread.join()
    consumer_thread.join()
    listener.stop()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if collected:
        print(f"[Main] 총 {len(collected)} 프레임 저장")
        data_np = np.array(collected)
        header = [f'{c}{i}' for i in range(21) for c in ['x', 'y', 'z']] + ['label']
        np.savetxt(filename, data_np, delimiter=',', header=','.join(header), fmt='%s', comments='')
        print(f"[Main] 저장 완료: {filename}")
    else:
        print("[Main] 저장할 데이터 없음")

if __name__ == '__main__':
    main()

