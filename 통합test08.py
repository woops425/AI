import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import pyautogui
import time
from collections import deque
from ultralytics import YOLO
import threading

# ===== 설정 =====
SEQUENCE_LENGTH       = 20
PREDICTION_THRESHOLD  = 0.80  # (참고) 이전 로직에서 사용. 현재는 히스테리시스 사용으로 대체됨.
TCN_PATH              = 'ai_motion_keyboard_GPU/assets/models/final_model_unified_v3.pt'
YOLO_PATH             = 'ai_motion_keyboard_GPU/assets/models/poker_tenkey(heavy).pt'
CORRECTION_INTERVAL_S = 5.0
INITIAL_DETECT_S      = 3.0
DEBOUNCE_MS           = 200    # 같은 손에서 연속 입력 방지 (밀리초)

# ---- 안정화 파라미터 (민감도 조절용) ----
PRED_ON_THRESH   = 0.8   # 켜질 때 임계(높일수록 덜 민감)
PRED_OFF_THRESH  = 0.7   # 꺼질 때 임계(낮출수록 빨리 꺼짐)
EMA_ALPHA        = 0.6   # 확률 EMA 계수(0~1). 낮추면 더 매끈, 반응성↓
MIN_ON_FRAMES    = 2      # ON 판정에 필요한 연속 프레임 수
MIN_OFF_FRAMES   = 2      # OFF 판정에 필요한 연속 프레임 수
REFRACTORY_MS    = 180    # 키 입력 후 불응기 (연타 방지)
# =================

# ===== 기능키 매핑 =====
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
    "right_Alt": "hangul",
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

# ===== TCN 정의 (기존 로딩 코드 유지) =====
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
    def forward(self, x):
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

# ===== 키보드 보정 상태 =====
corrected_boxes       = []
initial_key_positions = []
reference_initial     = {}
reference_current     = {}
reference_keys        = []
captured_initial      = False
last_correction_time  = 0.0
start_time            = time.time()
frame_buffer          = []

lock = threading.Lock()

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

# ===== YOLO 보정 스레드 =====
def yolo_correction_loop(cap):
    global corrected_boxes, last_correction_time, reference_current
    while True:
        if not captured_initial or keyboard_detector is None:
            time.sleep(1.0)
            continue
        now = time.time()
        if now - last_correction_time >= CORRECTION_INTERVAL_S:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)
            reference_current.clear()
            try:
                det = keyboard_detector.predict(frame, verbose=False)[0]
            except Exception as e:
                print(f"[WARN] YOLO 예외: {e}")
                last_correction_time = now
                time.sleep(0.1)
                continue
            for box in det.boxes:
                lbl = det.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2)/2, (y1 + y2)/2
                if lbl in reference_keys:
                    reference_current[lbl] = (cx, cy)
            if len(reference_current) >= 3:
                corrected = apply_transform_to_all_keys(reference_initial, reference_current, initial_key_positions)
                if corrected:
                    with lock:
                        corrected_boxes = corrected
                    print(f"[INFO] 보정 완료: {len(corrected_boxes)} boxes")
            else:
                print("[WARN] 보정 기준 부족 → 이전 값 유지")
            last_correction_time = now
        time.sleep(0.1)

# ===== 실시간 상태 (안정화 로직) =====
seq_data      = {'Right': deque(maxlen=SEQUENCE_LENGTH), 'Left': deque(maxlen=SEQUENCE_LENGTH)}
effect_xy     = {'Right': None, 'Left': None}

sm_prob       = {'Right': 0.0, 'Left': 0.0}   # EMA 확률
is_active     = {'Right': False, 'Left': False}
prev_active   = {'Right': False, 'Left': False}
on_cnt        = {'Right': 0, 'Left': 0}       # 연속 ON 프레임 카운터
off_cnt       = {'Right': 0, 'Left': 0}       # 연속 OFF 프레임 카운터
last_press_ms = {'Right': 0,  'Left': 0}      # 마지막 입력 시각(ms)

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
    if arr.ndim == 1:
        arr = arr.reshape(arr.shape[0], -1)
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
        prob = torch.softmax(out, dim=-1)[1].item()
    return prob

# ===== 카메라 =====
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[ERROR] 카메라 열기 실패")
    raise SystemExit

print("🟢 실행 중... 초기 키보드 자동 인식 대기 (약 3초) · ESC 종료")

# 보정 스레드 시작
correction_thread = threading.Thread(target=yolo_correction_loop, args=(cap,), daemon=True)
correction_thread.start()

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
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

            # (B) 초기 자동 인식 (메인 스레드에서 한 번 수행)
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
                    with lock:
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

            # (D) 손 처리 (MediaPipe) - 메인 루프는 항상 빠르게 동작
            res = hands.process(rgb)
            if res.multi_hand_landmarks:
                for i, lm in enumerate(res.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                    base_vec = extract_base_feature(lm.landmark)
                    hand_label = 'Right' if (res.multi_handedness and
                                             res.multi_handedness[i].classification[0].label == 'Right') else 'Left'
                    seq_data[hand_label].append(base_vec)
                    tip = lm.landmark[8]
                    effect_xy[hand_label] = (int(tip.x * w), int(tip.y * h))
            else:
                effect_xy['Right'] = None
                effect_xy['Left'] = None

            # (E) 예측 → EMA + 히스테리시스 + 연속 프레임 기준
            if model is not None:
                for hand_label in ('Right', 'Left'):
                    if len(seq_data[hand_label]) == SEQUENCE_LENGTH:
                        prob = predict_press(seq_data[hand_label]) or 0.0
                        # EMA
                        sm_prob[hand_label] = EMA_ALPHA * prob + (1.0 - EMA_ALPHA) * sm_prob[hand_label]

                        # 히스테리시스 + 연속 프레임
                        if not is_active[hand_label]:
                            if sm_prob[hand_label] >= PRED_ON_THRESH:
                                on_cnt[hand_label] += 1
                            else:
                                on_cnt[hand_label] = 0
                            if on_cnt[hand_label] >= MIN_ON_FRAMES:
                                is_active[hand_label] = True
                                off_cnt[hand_label] = 0
                        else:
                            if sm_prob[hand_label] <= PRED_OFF_THRESH:
                                off_cnt[hand_label] += 1
                            else:
                                off_cnt[hand_label] = 0
                            if off_cnt[hand_label] >= MIN_OFF_FRAMES:
                                is_active[hand_label] = False
                                on_cnt[hand_label] = 0
                    else:
                        # 시퀀스가 아직 안 찼으면 대기
                        is_active[hand_label] = False
                        on_cnt[hand_label] = off_cnt[hand_label] = 0

            # (F) 상승 에지 + 불응기 → 키 입력
            for hand_label in ('Right', 'Left'):
                if is_active[hand_label] and (not prev_active[hand_label]) and effect_xy[hand_label]:
                    now_ms = int(time.time() * 1000)
                    if (now_ms - last_press_ms[hand_label]) >= max(DEBOUNCE_MS, REFRACTORY_MS):
                        x_, y_ = effect_xy[hand_label]
                        with lock:
                            boxes_snapshot = corrected_boxes.copy()
                        for k in boxes_snapshot:
                            x0, y0, w0, h0 = int(k['x']), int(k['y']), int(k['width']), int(k['height'])
                            if x0 <= x_ <= x0 + w0 and y0 <= y_ <= y0 + h0:
                                mapped_key = key_name_map.get(k['label'], k['label'].lower())
                                try:
                                    pyautogui.press(mapped_key)
                                except Exception as e:
                                    print(f"[WARN] pyautogui.press 예외: {e}")
                                last_press_ms[hand_label] = now_ms
                                print(f"[{time.strftime('%H:%M:%S')}] {hand_label} Pressed {mapped_key} | p_ema={sm_prob[hand_label]:.2f}")
                                break
                prev_active[hand_label] = is_active[hand_label]

            # (G) 화면 표시 — p_ema & 상태
            line_y = 50
            for hand_label in ('Right', 'Left'):
                state_txt = "TYPING" if is_active[hand_label] else "NOT TYPING"
                cv2.putText(frame, f"{hand_label}: {state_txt}  p_ema={sm_prob[hand_label]:.2f}",
                            (10, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0,255,0) if is_active[hand_label] else (255,255,255), 2)
                line_y += 30

            if reference_keys:
                cv2.putText(frame, f"Reference: {', '.join(reference_keys)}",
                            (10, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                line_y += 25

            with lock:
                boxes_to_draw = corrected_boxes.copy()
            for k in boxes_to_draw:
                x0, y0, w0, h0 = int(k['x']), int(k['y']), int(k['width']), int(k['height'])
                color = (0,255,255) if k['label'] in reference_keys else (0,255,0)
                cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), color, 2)
                cv2.putText(frame, k['label'], (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            cv2.imshow("VK", frame)
            if key == 27: break  # ESC
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🛑 종료")
