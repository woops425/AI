import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import pyautogui
import time
from collections import deque
from ultralytics import YOLO

# —— 설정 값 —— 
SEQUENCE_LENGTH       = 20                   # TCN에 입력할 시퀀스 길이
PREDICTION_THRESHOLD  = 0.8                  # 분류 확률 임계값
TCN_PATH              = 'ai_motion_keyboard_GPU/assets/models/final_model_tcn_30ver_v4.pt' # TCN 모델 파일 경로
YOLO_PATH             = 'ai_motion_keyboard_GPU/assets/models/poker_tenkey(heavy).pt'  # YOLO 키보드 검출 모델 경로
CORRECTION_INTERVAL_S = 5.0                  # 키보드 보정 주기 (초)
INITIAL_DETECT_S      = 3.0                  # 시작 시 자동 인식 대기 시간 (초)
# ——————————————————

# ── 1. TCN 네트워크 구성 ──
class Chomp1d(nn.Module):
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp
    def forward(self, x):
        return x[:, :, :-self.chomp].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1  = nn.ReLU()
        self.drop1  = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2  = nn.ReLU()
        self.drop2  = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.chomp1(self.conv1(x))
        out = self.drop1(self.relu1(out))
        out = self.chomp2(self.conv2(out))
        out = self.drop2(self.relu2(out))
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)

class TCNModel(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout, out_features):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch   = num_inputs if i == 0 else num_channels[i-1]
            dilation = 2 ** i
            padding  = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation, padding, dropout)
            )
        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], out_features)

    def forward(self, x):
        y = self.tcn(x)
        y_last = y[:, :, -1]
        return self.linear(y_last)

# ── 2. TCN 모델 불러오기 ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    sd = torch.load(TCN_PATH, map_location=device)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    w0 = sd['tcn.0.conv1.weight_v']
    in_ch0, out0, k = w0.shape[1], w0.shape[0], w0.shape[2]
    w1 = sd['tcn.1.conv1.weight_v']
    out1 = w1.shape[0]
    out_feats = sd['linear.weight'].shape[0]

    typing_model = TCNModel(
        num_inputs=in_ch0,
        num_channels=[out0, out1],
        kernel_size=k,
        dropout=0.0,
        out_features=out_feats
    )
    typing_model.load_state_dict(sd)
    typing_model.to(device).eval()
    print(f"[INFO] TCN 모델 로드 성공: {TCN_PATH}")
except Exception as e:
    print(f"[ERROR] TCN 모델 로딩 실패: {e}")
    typing_model = None

# ── 3. YOLO 키보드 검출 모델 불러오기 ──
try:
    keyboard_detector = YOLO(YOLO_PATH)
    print(f"[INFO] YOLO 모델 로드 성공: {YOLO_PATH}")
except Exception as e:
    print(f"[ERROR] YOLO 모델 로딩 실패: {e}")
    keyboard_detector = None

# ── 4. MediaPipe Hands 초기화 ──
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ── 5. 키보드 레이아웃 자동 인식/보정 ──
corrected_boxes       = []
initial_key_positions = []
reference_initial     = {}
reference_current     = {}
reference_keys        = []
captured_initial      = False
last_correction_time  = 0.0
start_time            = time.time()
frame_buffer          = []

def auto_select_reference_keys(key_list):
    sorted_x = sorted(key_list, key=lambda k: k["center_x"])
    leftmost  = sorted_x[0]
    rightmost = sorted_x[-1]
    mid_candidates = sorted(key_list, key=lambda k: k["center_y"])
    mid_key = sorted(
        mid_candidates,
        key=lambda k: abs(k["center_x"] - (leftmost["center_x"] + rightmost["center_x"]) / 2)
    )[0]
    return [leftmost["label"], mid_key["label"], rightmost["label"]]

def apply_transform_to_all_keys(ref_init, ref_curr, key_positions):
    if len(reference_keys) < 3:
        return key_positions
    if not all(k in ref_init for k in reference_keys):
        return key_positions
    if not all(k in ref_curr for k in reference_keys):
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

# ── 6. 실시간 상태 변수 (양손) ──
seq_right = deque(maxlen=SEQUENCE_LENGTH)
seq_left  = deque(maxlen=SEQUENCE_LENGTH)   # 왼손은 x 미러 후 저장
curr_r, prev_r = 'WAITING', 'WAITING'
curr_l, prev_l = 'WAITING', 'WAITING'
effect_xy_right = None
effect_xy_left  = None

pyautogui.PAUSE    = 0
pyautogui.FAILSAFE = False

# ── 7. 웹캠 열기 ──
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[ERROR] 카메라를 열 수 없습니다.")
    exit()

print("🟢 시스템 실행 중... 초기 키보드 자동 인식 대기 중 (약 3초) · ESC 종료")

with mp_hands.Hands(
    max_num_hands=2,                 # ✅ 양손
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        key    = cv2.waitKey(1) & 0xFF
        now    = time.time()

        # (A) 초기 자동 인식 대기 & 안내
        if not captured_initial and (now - start_time) < INITIAL_DETECT_S:
            frame_buffer.append(frame.copy())
            cv2.putText(frame, "🔍 키보드 인식 중...", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("VK", frame)
            if key == 27:
                break
            continue

        # (B) 자동 초기 인식 시도
        if not captured_initial:
            if keyboard_detector is None:
                cv2.putText(frame, "YOLO 로드 실패 → 키보드 인식 불가", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("VK", frame)
                if key == 27:
                    break
                continue

            best_frame = frame_buffer[-1] if frame_buffer else frame
            det = keyboard_detector.predict(best_frame, verbose=False)[0]

            initial_key_positions.clear()
            reference_initial.clear()

            for box in det.boxes:
                lbl = det.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w0, h0 = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                initial_key_positions.append({
                    "label": lbl,
                    "x": x1, "y": y1,
                    "width": w0, "height": h0,
                    "center_x": cx, "center_y": cy
                })

            if len(initial_key_positions) >= 3:
                reference_keys = auto_select_reference_keys(initial_key_positions)
                for keyinfo in initial_key_positions:
                    if keyinfo["label"] in reference_keys:
                        reference_initial[keyinfo["label"]] = (keyinfo["center_x"], keyinfo["center_y"])

                corrected_boxes = [
                    {
                        "label": k["label"],
                        "x": k["x"], "y": k["y"],
                        "width": k["width"], "height": k["height"]
                    } for k in initial_key_positions
                ]

                captured_initial = True
                last_correction_time = now
                print(f"[INFO] 초기 인식 완료: {len(initial_key_positions)} keys")
                print(f"[INFO] 기준 키: {reference_keys}")
            else:
                print("[WARN] 키보드 인식 실패. 재시도 중...")
                start_time = time.time()
                frame_buffer.clear()

            cv2.imshow("VK", frame)
            if key == 27:
                break
            continue

        # (C) 주기 보정
        if keyboard_detector and (now - last_correction_time) >= CORRECTION_INTERVAL_S:
            reference_current.clear()
            det = keyboard_detector.predict(frame, verbose=False)[0]
            for box in det.boxes:
                lbl = det.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if lbl in reference_keys:
                    reference_current[lbl] = (cx, cy)

            if len(reference_current) >= 3:
                corrected_boxes = apply_transform_to_all_keys(
                    reference_initial, reference_current, initial_key_positions
                )
                print(f"[INFO] 보정 완료: {len(corrected_boxes)} boxes")
            else:
                print("[WARN] 보정 기준 키 부족. 이전 위치 유지")
            last_correction_time = now

        # (D) 손 랜드마크 & 특징 벡터(양손 처리)
        effect_xy_right = None
        effect_xy_left  = None

        res = hands.process(rgb)
        if res.multi_hand_landmarks and res.multi_handedness:
            for lm, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                hand = handedness.classification[0].label  # 'Right' or 'Left'
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                world = np.array([[p.x, p.y, p.z] for p in lm.landmark])
                wr    = world[0]
                rel   = world - wr

                # 왼손 → x축 미러링으로 오른손 좌표계로 변환
                rel_proc = rel.copy()
                if hand == 'Left':
                    rel_proc[:, 0] *= -1.0

                sc = np.linalg.norm(rel_proc[9])
                if sc > 1e-6:
                    fv = (rel_proc / sc).flatten()
                    if hand == 'Right':
                        seq_right.append(fv)
                        tip = lm.landmark[8]
                        effect_xy_right = (int(tip.x * w), int(tip.y * h))
                    else:
                        seq_left.append(fv)
                        tip = lm.landmark[8]
                        effect_xy_left  = (int(tip.x * w), int(tip.y * h))

        # (E) TCN 예측(양손 각각)
        def predict(seq_deque):
            if typing_model and len(seq_deque) == SEQUENCE_LENGTH:
                arr = np.array(seq_deque, dtype=np.float32).reshape(1, SEQUENCE_LENGTH, -1)
                t = torch.from_numpy(arr).permute(0, 2, 1).to(device)
                with torch.no_grad():
                    out = typing_model(t).squeeze()
                if out.numel() == 1:
                    prob = torch.sigmoid(out).item()
                else:
                    prob = torch.softmax(out, dim=-1)[1].item()
                return 'TYPING' if prob > PREDICTION_THRESHOLD else 'NOT TYPING'
            return 'WAITING'

        curr_r = predict(seq_right)
        curr_l = predict(seq_left)

        # (F) 상승 에지에서 키 입력 (오른손 우선, 없으면 왼손)
        if curr_r == 'TYPING' and prev_r != 'TYPING' and effect_xy_right:
            x_, y_ = effect_xy_right
            for k in corrected_boxes:
                x0, y0, w0, h0 = int(k['x']), int(k['y']), int(k['width']), int(k['height'])
                if x0 <= x_ <= x0 + w0 and y0 <= y_ <= y0 + h0:
                    pyautogui.press(k['label'])
                    print(f"[{time.strftime('%H:%M:%S')}] Pressed {k['label']} (Right)")
                    break

        if curr_l == 'TYPING' and prev_l != 'TYPING' and effect_xy_left:
            x_, y_ = effect_xy_left
            for k in corrected_boxes:
                x0, y0, w0, h0 = int(k['x']), int(k['y']), int(k['width']), int(k['height'])
                if x0 <= x_ <= x0 + w0 and y0 <= y_ <= y0 + h0:
                    pyautogui.press(k['label'])
                    print(f"[{time.strftime('%H:%M:%S')}] Pressed {k['label']} (Left)")
                    break

        prev_r, prev_l = curr_r, curr_l

        # (G) 상태 표시
        status = f"R:{curr_r}  L:{curr_l}"
        cv2.putText(frame, status, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if ('TYPING' in status) else (255, 255, 255), 2)

        if reference_keys:
            cv2.putText(frame, f"Reference: {', '.join(reference_keys)}",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        for k in corrected_boxes:
            x0, y0, w0, h0 = int(k['x']), int(k['y']), int(k['width']), int(k['height'])
            color = (0, 255, 255) if k['label'] in reference_keys else (0, 255, 0)
            cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), color, 2)
            cv2.putText(frame, k['label'], (x0, y0 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("VK", frame)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
print("🛑 프로그램 종료")