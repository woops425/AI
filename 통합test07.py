# 통합test06.py  — 기능키 매핑/입력 처리 적용 버전
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import pyautogui
import time
from collections import deque
from ultralytics import YOLO

# ===== 설정 =====
SEQUENCE_LENGTH       = 20
PREDICTION_THRESHOLD  = 0.80
TCN_PATH              = 'ai_motion_keyboard_GPU/assets/models/final_model_unified_v3.pt'
YOLO_PATH             = 'ai_motion_keyboard_GPU/assets/models/poker_tenkey(heavy).pt'   # 키보드 검출 모델
CORRECTION_INTERVAL_S = 5.0                        # 키보드 보정 주기(초)
INITIAL_DETECT_S      = 3.0                        # 시작 시 자동 인식 대기(초)
# =================

# ===== 기능키 매핑 (pyautogui 명칭으로 매핑) =====
key_name_map = {
    "Tlide": "`",
    "1": "1", "2": "2", "3": "3", "4": "4", "5": "5",
    "6": "6", "7": "7", "8": "8", "9": "9", "0": "0",
    "Subtract": "subtract",

    "Add": "add",
    "Backspace": "backspace",
    "Tab": "tab",
    "Q": "q", "W": "w", "E": "e", "R": "r", "T": "t",
    "Y": "y", "U": "u", "I": "i", "O": "o", "P": "p",
    "Bracket1": "[", "Bracket2": "]", "Backslash": "\\",
    "Caps Lock": "capslock",
    "A": "a", "S": "s", "D": "d", "F": "f", "G": "g",
    "H": "h", "J": "j", "K": "k", "L": "l",
    "Semi_colon": ";", "Apostrophe": "'",
    "Enter": "enter",
    "left_Shift": "shift", "right_Shift": "shift",
    "Z": "z", "X": "x", "C": "c", "V": "v", "B": "b", "N": "n", "M": "m",
    "Comma": ",", "Period": ".", "Slash": "/",
    "left_Ctrl": "ctrl", "right_Ctrl": "ctrl",
    "left_Win": "winleft", "right_Win": "winright",
    "left_Alt": "alt",
    "Space": "space",
    "right_Alt": "hangul",  # 한/영 전환
    "Menu": "menu",
    "Esc": "esc",
    "F1": "f1", "F2": "f2", "F3": "f3", "F4": "f4",
    "F5": "f5", "F6": "f6", "F7": "f7", "F8": "f8",
    "F9": "f9", "F10": "f10", "F11": "f11", "F12": "f12",
    "PrtSc": "printscreen", "Scroll": "scrolllock", "Pause": "pause",
    "Insert": "insert", "Home": "home", "PgUp": "pageup",
    "Delete": "delete", "End": "end", "PgDn": "pagedown",
    "up": "up", "left": "left", "down": "down", "right": "right"
}
# =================================================

# ===== TCN 정의 =====
class Chomp1d(nn.Module):
    def __init__(self, chomp): super().__init__(); self.chomp = chomp
    def forward(self, x): return x[:, :, :-self.chomp].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, kernel, padding=padding, dilation=dilation))
        self.chomp1= Chomp1d(padding); self.relu1=nn.ReLU(); self.drop1=nn.Dropout(dropout)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(out_ch, out_ch, kernel, padding=padding, dilation=dilation))
        self.chomp2= Chomp1d(padding); self.relu2=nn.ReLU(); self.drop2=nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.chomp1(self.conv1(x)); out = self.drop1(self.relu1(out))
        out = self.chomp2(self.conv2(out)); out = self.drop2(self.relu2(out))
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)

class TCNModel(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout, out_features):
        super().__init__()
        layers=[]
        for i, out_ch in enumerate(num_channels):
            in_ch = num_inputs if i==0 else num_channels[i-1]
            dilation = 2**i; padding = (kernel_size-1)*dilation
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, padding, dropout))
        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], out_features)
    def forward(self, x):  # x:(B,C,T)
        y = self.tcn(x); y_last = y[:, :, -1]
        return self.linear(y_last)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _first_block_conv1_weight(sd):
    cands=[]
    for k,v in sd.items():
        if k.startswith('tcn.') and (k.endswith('conv1.weight_v') or k.endswith('conv1.weight')):
            try: idx=int(k.split('.')[1])
            except: idx=9999
            cands.append((idx,k,v))
    if not cands: return None, None
    cands.sort(key=lambda x:x[0])
    return cands[0][1], cands[0][2]

def load_tcn_model_from_pt(path):
    try:
        sd = torch.load(path, map_location=device)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']

        k, w = _first_block_conv1_weight(sd)
        if w is None: raise RuntimeError('tcn.*.conv1.weight(_v) 없음')

        in_ch   = w.shape[1]
        kernel  = w.shape[2]
        blocks=[]
        for kk, vv in sd.items():
            if kk.endswith('conv1.weight_v') or kk.endswith('conv1.weight'):
                try: idx=int(kk.split('.')[1]); blocks.append((idx, vv.shape[0]))
                except: pass
        blocks.sort(key=lambda x:x[0])
        num_channels = [b[1] for b in blocks]

        lin_w = sd.get('linear.weight', None)
        if lin_w is None: raise RuntimeError('linear.weight 없음')
        out_features = lin_w.shape[0]

        model = TCNModel(in_ch, num_channels, kernel, dropout=0.0, out_features=out_features)
        model.load_state_dict(sd, strict=False)
        model.to(device).eval()
        model.expected_in_ch = in_ch
        model.out_features   = out_features
        print(f"[INFO] 통합 TCN 로드: {path} | in_ch={in_ch}, channels={num_channels}, kernel={kernel}, out={out_features}")
        return model
    except Exception as e:
        print(f"[ERROR] 모델 로딩 실패({path}): {e}")
        return None

model = load_tcn_model_from_pt(TCN_PATH)

# ===== YOLO 키보드 탐지 =====
try:
    keyboard_detector = YOLO(YOLO_PATH)
    print(f"[INFO] YOLO 로드 성공: {YOLO_PATH}")
except Exception as e:
    print(f"[ERROR] YOLO 로딩 실패: {e}")
    keyboard_detector = None

# ===== MediaPipe Hands =====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ===== 키보드 레이아웃 자동 인식/보정 =====
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
    leftmost, rightmost = sorted_x[0], sorted_x[-1]
    mid_key = sorted(key_list, key=lambda k: abs(k["center_x"] - (leftmost["center_x"] + rightmost["center_x"]) / 2))[0]
    return [leftmost["label"], mid_key["label"], rightmost["label"]]

def apply_transform_to_all_keys(ref_init, ref_curr, key_positions):
    if len(reference_keys) < 3: return key_positions
    if not all(k in ref_init for k in reference_keys): return key_positions
    if not all(k in ref_curr for k in reference_keys): return key_positions
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
            "width": key["width"], "height": key["height"]
        })
    return corrected

# ===== 실시간 상태 =====
seq_data   = {'Right': deque(maxlen=SEQUENCE_LENGTH), 'Left': deque(maxlen=SEQUENCE_LENGTH)}
curr_act   = {'Right': 'WAITING', 'Left': 'WAITING'}
prev_act   = {'Right': 'WAITING', 'Left': 'WAITING'}
effect_xy  = {'Right': None,       'Left': None}

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

def extract_base_feature(landmarks):
    world = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    wr = world[0]
    rel = world - wr
    sc = np.linalg.norm(rel[9])  # Index MCP까지 거리
    if sc < 1e-6: sc = 1.0
    return (rel / sc).flatten()  # (63,)

def predict_press(seq_vecs):
    if model is None: return None
    arr = np.asarray(seq_vecs, dtype=np.float32)
    if arr.shape[1] != getattr(model, 'expected_in_ch', arr.shape[1]):
        if arr.shape[1] < model.expected_in_ch:
            pad = np.zeros((arr.shape[0], model.expected_in_ch - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
            print(f"[WARN] 입력 채널 패딩: {arr.shape[1]}")
        else:
            arr = arr[:, :model.expected_in_ch]
            print(f"[WARN] 입력 채널 컷: {arr.shape[1]}")
    x = arr.reshape(1, SEQUENCE_LENGTH, -1)
    x = torch.from_numpy(x).permute(0, 2, 1).to(device)  # (B,C,T)
    with torch.no_grad():
        out = model(x).squeeze()
    if out.ndim == 0 or out.numel() == 1:
        prob = torch.sigmoid(out).item()
    else:
        prob = torch.softmax(out, dim=-1)[1].item()  # class1=눌림
    return prob

# ===== 카메라 =====
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[ERROR] 카메라 열기 실패")
    raise SystemExit

print("🟢 실행 중... 초기 키보드 자동 인식 대기 (약 3초) · ESC 종료")

with mp_hands.Hands(
    max_num_hands=2,  # 양손 처리
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        # (A) 초기 프레임 버퍼링
        if not captured_initial and (now - start_time) < INITIAL_DETECT_S:
            frame_buffer.append(frame.copy())
            cv2.putText(frame, "🔍 키보드 인식 중...", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.imshow("VK", frame)
            if key == 27: break
            continue

        # (B) 초기 자동 인식
        if not captured_initial:
            if keyboard_detector is None:
                cv2.putText(frame, "YOLO 로드 실패 → 키보드 인식 불가", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.imshow("VK", frame)
                if key == 27: break
                continue

            best_frame = frame_buffer[-1] if frame_buffer else frame
            det = keyboard_detector.predict(best_frame, verbose=False)[0]

            initial_key_positions.clear(); reference_initial.clear()
            for box in det.boxes:
                lbl = det.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w0, h0 = x2 - x1, y2 - y1
                cx, cy = (x1 + x2)/2, (y1 + y2)/2
                initial_key_positions.append({
                    "label": lbl, "x": x1, "y": y1, "width": w0, "height": h0,
                    "center_x": cx, "center_y": cy
                })

            if len(initial_key_positions) >= 3:
                reference_keys[:] = auto_select_reference_keys(initial_key_positions)
                for ki in initial_key_positions:
                    if ki["label"] in reference_keys:
                        reference_initial[ki["label"]] = (ki["center_x"], ki["center_y"])

                corrected_boxes = [
                    {"label": k["label"], "x": k["x"], "y": k["y"], "width": k["width"], "height": k["height"]}
                    for k in initial_key_positions
                ]
                captured_initial = True
                last_correction_time = now
                print(f"[INFO] 초기 인식 완료: {len(initial_key_positions)} keys, 기준 키={reference_keys}")
            else:
                print("[WARN] 키보드 인식 실패. 재시도...")
                start_time = time.time(); frame_buffer.clear()

            cv2.imshow("VK", frame)
            if key == 27: break
            continue

        # (C) 주기적 보정
        if keyboard_detector and (now - last_correction_time) >= CORRECTION_INTERVAL_S:
            reference_current.clear()
            det = keyboard_detector.predict(frame, verbose=False)[0]
            for box in det.boxes:
                lbl = det.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2)/2, (y1 + y2)/2
                if lbl in reference_keys:
                    reference_current[lbl] = (cx, cy)
            if len(reference_current) >= 3:
                corrected = apply_transform_to_all_keys(reference_initial, reference_current, initial_key_positions)
                if corrected: corrected_boxes = corrected
                print(f"[INFO] 보정 완료: {len(corrected_boxes)} boxes]")
            else:
                print("[WARN] 보정 기준 부족 → 이전 값 유지")
            last_correction_time = now

        # (D) 손 처리
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            for i, lm in enumerate(res.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                base_vec = extract_base_feature(lm.landmark)  # (63,)
                hand_label = 'Right' if (res.multi_handedness and
                                         res.multi_handedness[i].classification[0].label == 'Right') else 'Left'
                seq_data[hand_label].append(base_vec)
                tip = lm.landmark[8]
                effect_xy[hand_label] = (int(tip.x * w), int(tip.y * h))

        # (E) 각 손별 예측
        if model is not None:
            for hand_label in ('Right', 'Left'):
                if len(seq_data[hand_label]) == SEQUENCE_LENGTH:
                    prob = predict_press(seq_data[hand_label])
                    if prob is None:
                        curr_act[hand_label] = 'WAITING'
                    else:
                        curr_act[hand_label] = 'TYPING' if prob > PREDICTION_THRESHOLD else 'NOT TYPING'
                else:
                    curr_act[hand_label] = 'WAITING'

        # (F) 상승 에지에서 키 입력 — 기능키 매핑 적용
        for hand_label in ('Right', 'Left'):
            if curr_act[hand_label] == 'TYPING' and prev_act[hand_label] != 'TYPING' and effect_xy[hand_label]:
                x_, y_ = effect_xy[hand_label]
                for k in corrected_boxes:
                    x0, y0, w0, h0 = int(k['x']), int(k['y']), int(k['width']), int(k['height'])
                    if x0 <= x_ <= x0 + w0 and y0 <= y_ <= y0 + h0:
                        mapped_key = key_name_map.get(k['label'], k['label'].lower())
                        pyautogui.press(mapped_key)
                        print(f"[{time.strftime('%H:%M:%S')}] {hand_label} Pressed {mapped_key} (from '{k['label']}')")
                        break
            prev_act[hand_label] = curr_act[hand_label]

        # (G) 화면 표시
        line_y = 50
        for hand_label in ('Right', 'Left'):
            txt = f"{hand_label}: {curr_act[hand_label]}"
            color = (0,255,0) if curr_act[hand_label]=='TYPING' else (255,255,255)
            cv2.putText(frame, txt, (10, line_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            line_y += 30

        if reference_keys:
            cv2.putText(frame, f"Reference: {', '.join(reference_keys)}",
                        (10, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            line_y += 25

        for k in corrected_boxes:
            x0, y0, w0, h0 = int(k['x']), int(k['y']), int(k['width']), int(k['height'])
            color = (0,255,255) if k['label'] in reference_keys else (0,255,0)
            cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), color, 2)
            cv2.putText(frame, k['label'], (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        cv2.imshow("VK", frame)
        if key == 27: break  # ESC

cap.release()
cv2.destroyAllWindows()
print("🛑 종료")
