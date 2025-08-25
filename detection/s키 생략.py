import cv2
import numpy as np
import time
from ultralytics import YOLO

# 설정
corrected_keys = []
initial_key_positions = []
reference_keys = []
reference_initial = {}
reference_current = {}
captured_initial = False
last_correction_time = 0
correction_interval = 0.5  # 초 단위

# 기준 키 자동 선택 함수
def auto_select_reference_keys(key_list):
    sorted_keys = sorted(key_list, key=lambda k: k["center_x"])
    leftmost = sorted_keys[0]
    rightmost = sorted_keys[-1]
    mid_candidates = sorted(key_list, key=lambda k: k["center_y"])
    mid_key = sorted(mid_candidates, key=lambda k: abs(k["center_x"] - (leftmost["center_x"] + rightmost["center_x"]) / 2))[0]
    return [leftmost["label"], mid_key["label"], rightmost["label"]]

# 보정 함수
def apply_transform_to_all_keys(ref_init, ref_curr, key_positions):
    if len(ref_init) < 3 or len(ref_curr) < 3:
        return key_positions
    init_pts = np.array([ref_init[k] for k in reference_keys], dtype=np.float32)
    curr_pts = np.array([ref_curr[k] for k in reference_keys], dtype=np.float32)
    M = cv2.getAffineTransform(init_pts, curr_pts)
    corrected = []
    for key in key_positions:
        pt = np.array([[key['center_x'], key['center_y']]], dtype=np.float32)
        new_pt = cv2.transform(np.array([pt]), M)[0][0]
        corrected.append({
            "label": key["label"],
            "x": new_pt[0] - key["width"] / 2,
            "y": new_pt[1] - key["height"] / 2,
            "width": key["width"],
            "height": key["height"]
        })
    return corrected

# YOLO 모델 로드
yolo_model = YOLO("poker_tenkey(heavy).pt")

# 웹캠 시작
cap = cv2.VideoCapture(0)
print(" 시스템 실행 중... 초기 키보드 자동 인식 대기 중")

# 초기 자동 인식
initial_detection_duration = 3.0  # 초 단위
start_time = time.time()
frame_buffer = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # 자동 인식 수행 중
    if not captured_initial and current_time - start_time < initial_detection_duration:
        frame_buffer.append(frame.copy())
        cv2.putText(frame, "🔍 키보드 인식 중...", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Virtual Keyboard", frame)
        cv2.waitKey(1)
        continue

    # 인식 완료 시도
    if not captured_initial:
        print("🔍 키보드 자동 인식 시도 중...")
        best_frame = frame_buffer[-1] if frame_buffer else frame
        result = yolo_model.predict(best_frame, verbose=False)[0]
        initial_key_positions.clear()
        reference_initial.clear()
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            key_info = {
                "label": label,
                "x": x1,
                "y": y1,
                "width": w,
                "height": h,
                "center_x": cx,
                "center_y": cy
            }
            initial_key_positions.append(key_info)
        if len(initial_key_positions) >= 3:
            reference_keys = auto_select_reference_keys(initial_key_positions)
            print(f" 기준 키: {reference_keys}")
            for key in initial_key_positions:
                if key["label"] in reference_keys:
                    reference_initial[key["label"]] = (key["center_x"], key["center_y"])
            captured_initial = True
            print(f" {len(initial_key_positions)}개 키 인식 완료")
        else:
            print(" 키보드 인식 실패. 프레임 수 부족 또는 키 미탐지")
            start_time = time.time()  # 재시도
            frame_buffer.clear()
        continue

    # 보정 수행
    if current_time - last_correction_time >= correction_interval:
        result = yolo_model.predict(frame, verbose=False)[0]
        reference_current.clear()
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if label in reference_keys:
                reference_current[label] = (cx, cy)

        if len(reference_current) >= 3:
            corrected_keys = apply_transform_to_all_keys(reference_initial, reference_current, initial_key_positions)
            last_correction_time = current_time
        else:
            print(" 보정 기준 키 부족으로 기존 위치 유지")

    # 시각화 출력
    cv2.putText(frame, f"Reference Keys: {', '.join(reference_keys)}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    for key in corrected_keys:
        x, y, w, h = int(key['x']), int(key['y']), int(key['width']), int(key['height'])
        color = (0, 255, 255) if key['label'] in reference_keys else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, key['label'], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("Virtual Keyboard", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(" 시스템 종료")