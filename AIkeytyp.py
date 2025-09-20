import sys
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
import cpuinfo

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSplashScreen, QMessageBox
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QObject, Signal, QThread

# ===== 설정 =====
SEQUENCE_LENGTH       = 20
TCN_PATH              = 'ai_motion_keyboard_GPU/assets/models/final_model_unified_v3.pt'
YOLO_PATH             = 'ai_motion_keyboard_GPU/assets/models/poker_tenkey(heavy).pt'
DEBOUNCE_MS           = 200   # 같은 손에서 연속 입력 방지 (밀리초)

# ▼▼▼ [추가] 카메라 선택 스위치 ▼▼▼
USE_NETWORK_CAM = False  # True: 네트워크 카메라, False: 로컬 웹캠(0번)
NETWORK_CAM_URL = "http://192.168.219.100:4747/video?1280x720"

# ---- 안정화 파라미터 (민감도 조절용) ----
PRED_ON_THRESH   = 0.75  # 켜질 때 임계(높일수록 덜 민감)
PRED_OFF_THRESH  = 0.70  # 꺼질 때 임계(낮출수록 빨리 꺼짐)
EMA_ALPHA        = 0.60  # 확률 EMA 계수(0~1). 낮추면 더 매끈, 반응성↓
MIN_ON_FRAMES    = 2     # ON 판정에 필요한 연속 프레임 수
MIN_OFF_FRAMES   = 2     # OFF 판정에 필요한 연속 프레임 수
REFRACTORY_MS    = 180   # 키 입력 후 불응기 (연타 방지)

# ---- [NEW] 키보드 자동 인식 설정 ----
VALID_KEY_COUNTS      = [87, 61] # 유효 키보드 키 개수 목록
MOVEMENT_THRESHOLD    = 4.5  # 평균 픽셀 변화량 (낮을수록 민감)
STABLE_TIME_REQUIRED  = 1.0  # 안정 상태 유지 필요 시간 (초)
CORRECTION_INTERVAL_S = 5.0  # 주기적 위치 보정 간격 (초)
RESET_KEY_COUNT_THRESHOLD = 1 # 이 개수 미만으로 감지되면 즉시 재인식
# =================


# ===== 기능키 매핑 =====
key_name_map = {
    "Tlide": "`", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9", "0": "0", "Subtract": "subtract", "Add": "add",
    "Backspace": "backspace", "Tab": "tab", "Q": "q", "W": "w", "E": "e", "R": "r", "T": "t", "Y": "y", "U": "u", "I": "i", "O": "o", "P": "p",
    "Bracket1": "[", "Bracket2": "]", "Backslash": "\\", "Caps Lock": "capslock", "A": "a", "S": "s", "D": "d", "F": "f", "G": "g", "H": "h", "J": "j", "K": "k", "L": "l",
    "Semi_colon": ";", "Apostrophe": "'", "Enter": "enter", "left_Shift": "shift", "right_Shift": "shift", "Z": "z", "X": "x", "C": "c", "V": "v", "B": "b", "N": "n", "M": "m",
    "Comma": ",", "Period": ".", "Slash": "/", "left_Ctrl": "ctrl", "right_Ctrl": "ctrl", "left_Win": "winleft", "right_Win": "winright", "left_Alt": "alt",
    "Space": "space", "right_Alt": "hangul", "Menu": "menu", "Esc": "esc", "F1": "f1", "F2": "f2", "F3": "f3", "F4": "f4", "F5": "f5", "F6": "f6", "F7": "f7", "F8": "f8",
    "F9": "f9", "F10": "f10", "F11": "f11", "F12": "f12", "PrtSc": "printscreen", "Scroll": "scrolllock", "Pause": "pause", "Insert": "insert", "Home": "home", "PgUp": "pageup",
    "Delete": "delete", "End": "end", "PgDn": "pagedown", "up": "up", "left": "left", "down": "down", "right": "right"
}

# ===== [NEW] 화면 표시용 이름 맵 =====
display_name_map = {
    "left_Shift": "LSH", "right_Shift": "RSH",
    "left_Ctrl": "LC", "right_Ctrl": "RC",
    "left_Win": "LW", "right_Win": "RW",
    "left_Alt": "LA", "right_Alt": "RA",
    "Caps Lock": "CAPS",
    "Enter": "ENT",
    "Tab": "TAB",
    "Backspace": "BSPC",

    # ▼▼▼ 기호/특수키 매핑 추가 ▼▼▼
    "Tlide": "`",
    "Subtract": "-",
    "Add": "+",
    "Bracket1": "[",
    "Bracket2": "]",
    "Backslash": "\\",
    "Semi_colon": ";",
    "Apostrophe": "'",
    "Comma": ",",
    "Period": ".",
    "Slash": "/",
    # ... (원하는 다른 키들도 추가 가능) ...
}

# ===== TCN 모델 클래스 정의 =====
class Chomp1d(nn.Module):
    def __init__(self, chomp): super().__init__(); self.chomp = chomp
    def forward(self, x): return x[:, :, :-self.chomp].contiguous()
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, padding, dropout):
        super().__init__();self.conv1=nn.utils.weight_norm(nn.Conv1d(in_ch,out_ch,kernel,padding=padding,dilation=dilation));self.chomp1=Chomp1d(padding);self.relu1=nn.ReLU();self.drop1=nn.Dropout(dropout);self.conv2=nn.utils.weight_norm(nn.Conv1d(out_ch,out_ch,kernel,padding=padding,dilation=dilation));self.chomp2=Chomp1d(padding);self.relu2=nn.ReLU();self.drop2=nn.Dropout(dropout);self.downsample=nn.Conv1d(in_ch,out_ch,1) if in_ch!=out_ch else None;self.relu=nn.ReLU()
    def forward(self,x):
        out=self.chomp1(self.conv1(x));out=self.drop1(self.relu1(out));out=self.chomp2(self.conv2(out));out=self.drop2(self.relu2(out));residual=x if self.downsample is None else self.downsample(x);return self.relu(out+residual)
class TCNModel(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout, out_features):
        super().__init__();layers=[];
        for i,out_ch in enumerate(num_channels):
            in_ch=num_inputs if i==0 else num_channels[i-1];dilation=2**i;padding=(kernel_size-1)*dilation;layers.append(TemporalBlock(in_ch,out_ch,kernel_size,dilation,padding,dropout))
        self.tcn=nn.Sequential(*layers);self.linear=nn.Linear(num_channels[-1],out_features)
    def forward(self,x): y=self.tcn(x);y_last=y[:,:,-1];return self.linear(y_last)

# ===== 하드웨어 상태 확인 =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_name, gpu_name = "", ""
print("\n" + "="*30); print("   HARDWARE STATUS CHECK"); print("="*30)
try:
    cpu_name = cpuinfo.get_cpu_info()['brand_raw']
    print(f"✅ CPU Detected: {cpu_name}")
except Exception:
    cpu_name = "CPU (Unknown)"
if device.type == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU (CUDA) is available!: {gpu_name}")
else:
    gpu_name = "N/A (CPU Mode)"
print("="*30 + "\n")

# ===== VideoStream 클래스 (들여쓰기 수정) =====
class VideoStream:
    def __init__(self, src=0):
        # src 타입에 따라 다르게 카메라를 엽니다
        if isinstance(src, int):
            # src가 숫자(0, 1 등)이면 로컬 웹캠으로 간주하고 CAP_DSHOW를 사용
            self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            print(f"[INFO] Local webcam ({src}) starting with DSHOW backend.")
        else:
            # src가 문자열이면 네트워크 카메라로 간주
            self.stream = cv2.VideoCapture(src)
            print(f"[INFO] Network camera starting: {src}")

        if not self.stream.isOpened(): raise IOError(f"Cannot open video stream: {src}")
        (self.ret, self.frame) = self.stream.read()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # 이 메서드는 __init__과 동일한 수준으로 들여쓰기 되어야 합니다.
        while not self.stopped:
            (self.ret, self.frame) = self.stream.read()

    def read(self):
        # 이 메서드는 __init__과 동일한 수준으로 들여쓰기 되어야 합니다.
        return self.frame

    def stop(self):
        # 이 메서드는 __init__과 동일한 수준으로 들여쓰기 되어야 합니다.
        self.stopped = True
        self.thread.join()
        self.stream.release()

# ===== Worker 클래스 =====
class Worker(QObject):
    finished = Signal()
    new_frame = Signal(np.ndarray)
    update_status = Signal(str)
    reference_keys_updated = Signal(str)
    preset_updated_signal = Signal(str) # 포맷된 전체 문자열을 전달

    def __init__(self, vs):
        super().__init__()
        self.vs = vs
        self.running = True
        # --- 모든 AI 모델과 변수들을 클래스 멤버로 초기화 ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_tcn_model_from_pt(TCN_PATH, self.device)
        self.keyboard_detector = YOLO(YOLO_PATH)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.last_pressed_info = {'key_label': None, 'time': 0}

        # ▼▼▼ 프리셋 파라미터를 self 변수로 초기화 ▼▼▼
        self.PRED_ON_THRESH = 0
        self.PRED_OFF_THRESH = 0
        self.EMA_ALPHA = 0
        self.REFRACTORY_MS = 0

        self.corrected_boxes = []; self.initial_key_positions = []; self.reference_keys = []; self.reference_initial = {}; self.reference_current = {}
        self.captured_initial = False; self.last_correction_time = 0; self.lock = threading.Lock()
        self.prev_gray = None; self.stable_start_time = None
        self.seq_data = {'Right': deque(maxlen=SEQUENCE_LENGTH), 'Left': deque(maxlen=SEQUENCE_LENGTH)}
        self.effect_xy = {'Right': None, 'Left': None}; self.sm_prob = {'Right': 0.0, 'Left': 0.0}
        self.is_active = {'Right': False, 'Left': False}; self.prev_active = {'Right': False, 'Left': False}
        self.on_cnt = {'Right': 0, 'Left': 0}; self.off_cnt = {'Right': 0, 'Left': 0}; self.last_press_ms = {'Right': 0, 'Left': 0}

        # ▼▼▼ 입력 활성화 상태 변수 추가 (기본값: True) ▼▼▼
        self.input_enabled = True

        # ▼▼▼ 프리셋 파라미터 딕셔너리 추가 ▼▼▼
        self.presets = {
            "Sensitive": {"PRED_ON_THRESH": 0.65, "PRED_OFF_THRESH": 0.60, "EMA_ALPHA": 0.70, "REFRACTORY_MS": 180},
            "Balanced": {"PRED_ON_THRESH": 0.75, "PRED_OFF_THRESH": 0.70, "EMA_ALPHA": 0.60, "REFRACTORY_MS": 180},
            "Stable": {"PRED_ON_THRESH": 0.80, "PRED_OFF_THRESH": 0.70, "EMA_ALPHA": 0.60, "REFRACTORY_MS": 180},
            #"Very Stable": {"PRED_ON_THRESH": 0.90, "PRED_OFF_THRESH": 0.85,"EMA_ALPHA": 0.30, "REFRACTORY_MS": 300}
        }

    # Worker 클래스 내부 (예: __init__ 메서드 다음)
    def set_input_enabled(self, enabled):
        """UI 스레드로부터 신호를 받아 입력 활성화 상태를 변경하는 슬롯"""
        self.input_enabled = enabled
        print(f"[INFO] Keyboard input {'ENABLED' if enabled else 'DISABLED'}")

    def force_rerecognize(self):
        """UI로부터 신호를 받아 키보드 재인식을 강제하는 슬롯"""
        print("[INFO] Manual keyboard re-recognition triggered!")
        self.captured_initial = False
        self.stable_start_time = None
        self.reference_keys_updated.emit("Reference Keys: N/A")

    def apply_preset(self, preset_name):
        """UI로부터 신호를 받아 타이핑 민감도 프리셋을 적용하는 슬롯"""
        if preset_name in self.presets:
            print(f"[INFO] Applying preset: {preset_name}")
            preset_values = self.presets[preset_name]

            # 전역 변수가 아닌 클래스 멤버 변수를 직접 수정
            self.PRED_ON_THRESH = preset_values["PRED_ON_THRESH"]
            self.PRED_OFF_THRESH = preset_values["PRED_OFF_THRESH"]
            self.EMA_ALPHA = preset_values["EMA_ALPHA"]
            self.REFRACTORY_MS = preset_values["REFRACTORY_MS"]

            # ▼▼▼ 변경된 프리셋 정보를 UI로 다시 보내는 신호 발생 ▼▼▼
            info_text = (f"Preset: {preset_name}\n"
                        f"ON:{self.PRED_ON_THRESH:.2f} OFF:{self.PRED_OFF_THRESH:.2f} "
                        f"EMA:{self.EMA_ALPHA:.2f}")
            self.preset_updated_signal.emit(info_text)

    # ... (모든 헬퍼 함수들은 이전과 동일하게 여기에 위치) ...
    def _first_block_conv1_weight(self, sd):
        cands=[];
        for k,v in sd.items():
            if k.startswith('tcn.') and (k.endswith('conv1.weight_v') or k.endswith('conv1.weight')):
                try:idx=int(k.split('.')[1])
                except:idx=9999
                cands.append((idx,k,v))
        if not cands:return None,None
        cands.sort(key=lambda x:x[0]);return cands[0][1],cands[0][2]
    def load_tcn_model_from_pt(self, path, device):
        try:
            sd=torch.load(path,map_location=device);
            if isinstance(sd,dict) and 'state_dict' in sd:sd=sd['state_dict']
            k,w=self._first_block_conv1_weight(sd);
            if w is None:raise RuntimeError('tcn.*.conv1.weight(_v) 없음')
            in_ch=w.shape[1];kernel=w.shape[2];blocks=[];
            for kk,vv in sd.items():
                if kk.endswith('conv1.weight_v') or kk.endswith('conv1.weight'):
                    try:idx=int(kk.split('.')[1]);blocks.append((idx,vv.shape[0]))
                    except:pass
            blocks.sort(key=lambda x:x[0]);num_channels=[b[1] for b in blocks];lin_w=sd.get('linear.weight',None);
            if lin_w is None:raise RuntimeError('linear.weight 없음')
            out_features=lin_w.shape[0];model=TCNModel(in_ch,num_channels,kernel,dropout=0.0,out_features=out_features);model.load_state_dict(sd,strict=False);model.to(device).eval();model.expected_in_ch=in_ch;model.out_features=out_features;print(f"[INFO] 통합 TCN 로드: {path} | in_ch={in_ch}, channels={num_channels}, kernel={kernel}, out={out_features}");return model
        except Exception as e:print(f"[ERROR] 모델 로딩 실패({path}): {e}");return None
    def auto_select_reference_keys(self, key_list):
        sorted_keys=sorted(key_list,key=lambda k:k["center_x"]);leftmost=sorted_keys[0];rightmost=sorted_keys[-1];mid_candidates=sorted(key_list,key=lambda k:k["center_y"]);mid_key=sorted(mid_candidates,key=lambda k:abs(k["center_x"]-(leftmost["center_x"]+rightmost["center_x"])/2))[0];return [leftmost["label"],mid_key["label"],rightmost["label"]]
    def apply_transform_to_all_keys(self, ref_init, ref_curr, key_positions, reference_keys):
        if len(ref_init)<3 or len(ref_curr)<3:return key_positions
        init_pts=np.array([ref_init[k] for k in reference_keys],dtype=np.float32);curr_pts=np.array([ref_curr[k] for k in reference_keys],dtype=np.float32)
        M=cv2.getAffineTransform(init_pts,curr_pts);corrected=[]
        for key in key_positions:
            pt=np.array([[key['center_x'],key['center_y']]],dtype=np.float32);new_pt=cv2.transform(np.array([pt]),M)[0][0];corrected.append({"label":key["label"],"x":new_pt[0]-key["width"]/2,"y":new_pt[1]-key["height"]/2,"width":key["width"],"height":key["height"]})
        return corrected
    def extract_base_feature(self, landmarks):
        world=np.array([[p.x,p.y,p.z] for p in landmarks],dtype=np.float32);wr=world[0];rel=world-wr;sc=np.linalg.norm(rel[9]);
        if sc<1e-6:sc=1.0
        return (rel/sc).flatten()
    def predict_press(self, seq_vecs, model, device):
        if model is None:return None
        arr=np.asarray(seq_vecs,dtype=np.float32)
        if arr.ndim==1:arr=arr.reshape(arr.shape[0],-1)
        if arr.shape[1]!=getattr(model,'expected_in_ch',arr.shape[1]):
            if arr.shape[1]<model.expected_in_ch:pad=np.zeros((arr.shape[0],model.expected_in_ch-arr.shape[1]),dtype=np.float32);arr=np.concatenate([arr,pad],axis=1)
            else:arr=arr[:,:model.expected_in_ch]
        x=arr.reshape(1,SEQUENCE_LENGTH,-1);x=torch.from_numpy(x).permute(0,2,1).to(device)
        with torch.no_grad():out=model(x).squeeze()
        if out.ndim==0 or out.numel()==1:prob=torch.sigmoid(out).item()
        else:prob=torch.softmax(out,dim=-1)[1].item()
        return prob

    # [NEW] 키보드 초기 인식 전용 메서드
    def setup_keyboard(self):
        while not self.captured_initial and self.running:
            frame = self.vs.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            movement_amount = None

            if self.prev_gray is not None:
                diff = cv2.absdiff(self.prev_gray, gray)
                movement_amount = np.mean(diff)

                if movement_amount < MOVEMENT_THRESHOLD:
                    if self.stable_start_time is None: self.stable_start_time = time.time()
                    elif time.time() - self.stable_start_time >= STABLE_TIME_REQUIRED:
                        print("📷 안정 상태 감지. 키보드 자동 인식을 시도합니다...")
                        det = self.keyboard_detector.predict(frame, verbose=False, device=self.device)[0]
                        temp_key_positions = []
                        for box in det.boxes:
                            lbl=det.names[int(box.cls[0])]; x1,y1,x2,y2=map(int,box.xyxy[0]); w0,h0=x2-x1,y2-y1; cx,cy=(x1+x2)/2,(y1+y2)/2
                            temp_key_positions.append({"label":lbl,"x":x1,"y":y1,"width":w0,"height":h0,"center_x":cx,"center_y":cy})

                        detected_count = len(temp_key_positions)
                        if detected_count in VALID_KEY_COUNTS:
                            self.initial_key_positions = temp_key_positions
                            self.reference_keys = self.auto_select_reference_keys(self.initial_key_positions)
                            for ki in self.initial_key_positions:
                                if ki["label"] in self.reference_keys: self.reference_initial[ki["label"]] = (ki["center_x"], ki["center_y"])
                            with self.lock:
                                self.corrected_boxes = [{"label":k["label"],"x":k["x"],"y":k["y"],"width":k["width"],"height":k["height"]} for k in self.initial_key_positions]

                            self.captured_initial = True
                            self.last_correction_time = time.time()
                            status_text = f"Reference Keys: {', '.join(self.reference_keys)}"
                            print(f"✅ {detected_count}개 키 인식 성공! 기준 키: {self.reference_keys}")
                            self.reference_keys_updated.emit(status_text)
                        else:
                            print(f"❌ 인식 실패 (감지된 키 개수: {detected_count}).")
                            self.stable_start_time = None
                else:
                    self.stable_start_time = None

            self.prev_gray = gray.copy()
            cv2.putText(frame, "Waiting for keyboard to stabilize...", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if movement_amount is not None: cv2.putText(frame, f"Movement: {movement_amount:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            self.new_frame.emit(frame) # UI에 현재 상태 프레임 전송

            # ▼▼▼ [추가] UI 이벤트를 처리할 시간을 줍니다. ▼▼▼
            QApplication.processEvents()


    # [NEW] 실시간 타이핑 처리 전용 메서드
    def main_loop(self):
        with self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while self.running:
                # ▼▼▼ [수정] 이 if문을 추가하여 재인식 신호를 확인합니다 ▼▼▼
                if not self.captured_initial:
                    break # 재인식 상태가 되면 main_loop를 탈출

                frame = self.vs.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                now = time.time()
                now_ms = int(now * 1000)

                # ... (이하 모든 실시간 처리 로직) ...
                if now - self.last_correction_time >= CORRECTION_INTERVAL_S:
                    det = self.keyboard_detector.predict(frame, verbose=False, device=self.device)[0]
                    if len(det.boxes) < RESET_KEY_COUNT_THRESHOLD:
                        print("🚨 키보드 감지 불가! 즉시 재인식을 시작합니다.")
                        self.captured_initial=False
                        self.reference_keys_updated.emit("Reference Keys: N/A")
                        return # main_loop를 종료하고 run 메서드로 돌아가 setup_keyboard를 다시 실행하게 함
                    else:
                        self.reference_current.clear()
                        for box in det.boxes:
                            lbl=det.names[int(box.cls[0])]; x1,y1,x2,y2=map(int,box.xyxy[0]); cx,cy=(x1+x2)/2,(y1+y2)/2
                            if lbl in self.reference_keys: self.reference_current[lbl]= (cx,cy)
                        if len(self.reference_current) >= 3:
                            corrected = self.apply_transform_to_all_keys(self.reference_initial, self.reference_current, self.initial_key_positions, self.reference_keys)
                            if corrected:
                                with self.lock: self.corrected_boxes = corrected
                        else:
                            print("[WARN] 보정 기준 부족 → 이전 값 유지")
                    self.last_correction_time = now

                res = hands.process(rgb)
                # ... (이하 손 처리, 예측, 키 입력, 그리기 로직은 이전과 동일) ...
                if res.multi_hand_landmarks:
                    for i, lm in enumerate(res.multi_hand_landmarks):
                        self.mp_drawing.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)
                        base_vec = self.extract_base_feature(lm.landmark)
                        hand_label = 'Right' if (res.multi_handedness and res.multi_handedness[i].classification[0].label == 'Right') else 'Left'
                        self.seq_data[hand_label].append(base_vec)
                        tip = lm.landmark[8]
                        self.effect_xy[hand_label] = (int(tip.x * w), int(tip.y * h))
                else:
                    self.effect_xy['Right'] = self.effect_xy['Left'] = None

                if self.model is not None:
                    for hand_label in ('Right', 'Left'):
                        if len(self.seq_data[hand_label]) == SEQUENCE_LENGTH:
                            prob = self.predict_press(self.seq_data[hand_label], self.model, self.device) or 0.0
                            self.sm_prob[hand_label] = EMA_ALPHA * prob + (1.0 - EMA_ALPHA) * self.sm_prob[hand_label]
                            if not self.is_active[hand_label]:
                                if self.sm_prob[hand_label] >= self.PRED_ON_THRESH: self.on_cnt[hand_label] += 1
                                else: self.on_cnt[hand_label] = 0
                                if self.on_cnt[hand_label] >= MIN_ON_FRAMES: self.is_active[hand_label] = True; self.off_cnt[hand_label] = 0
                            else:
                                if self.sm_prob[hand_label] <= self.PRED_OFF_THRESH: self.off_cnt[hand_label] += 1
                                else: self.off_cnt[hand_label] = 0
                                if self.off_cnt[hand_label] >= MIN_OFF_FRAMES: self.is_active[hand_label] = False; self.on_cnt[hand_label] = 0
                        else:
                            self.is_active[hand_label] = False
                            self.on_cnt[hand_label] = self.off_cnt[hand_label] = 0

                for hand_label in ('Right', 'Left'):
                    if self.is_active[hand_label] and (not self.prev_active[hand_label]) and self.effect_xy[hand_label]:
                        if (now_ms - self.last_press_ms[hand_label]) >= max(DEBOUNCE_MS, self.REFRACTORY_MS):
                            x_, y_ = self.effect_xy[hand_label]
                            with self.lock: boxes_snapshot = self.corrected_boxes.copy()
                            for k in boxes_snapshot:
                                x0,y0,w0,h0=int(k['x']),int(k['y']),int(k['width']),int(k['height'])
                                if x0<=x_<=x0+w0 and y0<=y_<=y0+h0:
                                    mapped_key=key_name_map.get(k['label'],k['label'].lower())
                                    # ▼▼▼ 입력 활성화 상태일 때만 pyautogui.press 실행 ▼▼▼
                                    if self.input_enabled:
                                        pyautogui.press(mapped_key)
                                    now = time.time() # 현재 시간을 다시 가져옴
                                    self.last_pressed_info = {'key_label': k['label'], 'time': now}
                                    self.last_press_ms[hand_label] = now_ms
                                    print(f"[{time.strftime('%H:%M:%S')}] {hand_label} Pressed {mapped_key} | p_ema={self.sm_prob[hand_label]:.2f}")
                                    break
                    self.prev_active[hand_label] = self.is_active[hand_label]

                # (G) 화면 표시
                right_state = "TYPING" if self.is_active['Right'] else "NOT TYPING"
                right_text = f"Right: {right_state} (p_ema={self.sm_prob['Right']:.2f})"
                left_state = "TYPING" if self.is_active['Left'] else "NOT TYPING"
                left_text = f"Left: {left_state} (p_ema={self.sm_prob['Left']:.2f})"
                combined_status = f"{right_text} | {left_text}"
                self.update_status.emit(combined_status)

                # ▼▼▼ [수정] line_y 정의 및 기준 키 그리기 로직 (화면 표시의 시작 부분으로 이동) ▼▼▼
                line_y=50
                if self.reference_keys:
                    # 이 텍스트는 이제 Worker 스레드가 아닌 MainWindow의 상태바에 표시됩니다.
                    # cv2.putText(frame,f"Reference: {', '.join(reference_keys)}",(10,line_y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
                    pass # 나중에 다른 정보를 표시할 수 있도록 남겨둡니다.

                with self.lock:
                    boxes_to_draw = self.corrected_boxes.copy()

                # ▼▼▼ [추가] 키 입력 시각적 피드백 로직 ▼▼▼
                # 키가 눌렸을 때 잠깐 동안 색을 변경하기 위한 변수가 필요합니다.
                # 이 변수들은 Worker의 __init__에 추가되어야 하지만, 설명을 위해 여기에 로직을 먼저 보여드립니다.
                # (실제 구현 시에는 __init__에 self.last_pressed_info = {'key': None, 'time': 0} 추가 필요)

                for k in boxes_to_draw:
                    x0,y0,w0,h0=int(k['x']),int(k['y']),int(k['width']),int(k['height'])

                    # 기본 색상 설정
                    color=(0,255,255) if k['label'] in self.reference_keys else (0,255,0)

                    # 키 입력 피드백 확인
                    # (self.last_pressed_info는 나중에 __init__에 추가해야 합니다)
                    if hasattr(self, 'last_pressed_info') and k['label'] == self.last_pressed_info['key_label']:
                        if now - self.last_pressed_info['time'] < 0.2: # 0.2초 동안 노란색으로 표시
                            color = (255, 0, 0) # 파란색

                    cv2.rectangle(frame,(x0,y0),(x0+w0,y0+h0),color,1)

                    display_label = display_name_map.get(k['label'], k['label'])
                    cv2.putText(frame, display_label, (x0 + 5, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # ▼▼▼ [추가] 핑거팁 커서 시각화 로직 ▼▼▼
                for hand_label in ('Right', 'Left'):
                    if self.effect_xy[hand_label]:
                        center_coordinates = self.effect_xy[hand_label]
                        radius = 10
                        color = (0, 0, 255) # 빨간색
                        thickness = 0 # 원 내부를 비움
                        cv2.circle(frame, center_coordinates, radius, color, thickness)

                self.new_frame.emit(frame)

                # ▼▼▼ [추가] UI 이벤트를 처리할 시간을 줍니다. ▼▼▼
                QApplication.processEvents()

    # [NEW] run 메서드의 새로운 구조
    def run(self):
        self.apply_preset("Balanced")
        while self.running:
            self.setup_keyboard()
            if not self.running: break # 사용자가 setup 도중 종료 시
            self.main_loop()
        self.finished.emit()

    def stop(self):
        self.running = False

# ===== MainWindow 클래스 (UI 레이아웃 확장) =====
class MainWindow(QMainWindow):
    # ▼▼▼ Worker에게 보낼 신호 추가 (bool 타입의 데이터를 전달) ▼▼▼
    toggle_input_signal = Signal(bool)
    rerecognize_signal = Signal()    # 재인식 신호 추가
    apply_preset_signal = Signal(str) # 프리셋 적용 신호 추가

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Virtual Keyboard")
        self.setGeometry(100, 100, 1366, 768) # 창 크기를 약간 더 넓게 설정
        self.center() # 창을 화면 중앙으로 이동시키는 메서드 호출

        # --- [추가] 다크/라이트 모드 스타일시트 정의 ---
        self.light_mode_style = """
            QMainWindow { background-color: #f0f0f0; }
            QLabel { color: black; }
            QPushButton { background-color: #e0e0e0; color: black; border: 1px solid #b0b0b0; padding: 5px; }
            QPushButton:hover { background-color: #d0d0d0; }
            QMenuBar { background-color: #e0e0e0; color: black; }
            QMenuBar::item:selected { background-color: #d0d0d0; }
            QMenu { background-color: #f0f0f0; color: black; }
            QMenu::item:selected { background-color: #d0d0d0; }
        """
        self.dark_mode_style = """
            QMainWindow { background-color: #2b2b2b; }
            QLabel { color: white; }
            QPushButton { background-color: #505050; color: white; border: 1px solid #606060; padding: 5px; }
            QPushButton:hover { background-color: #606060; }
            QMenuBar { background-color: #3c3c3c; color: white; }
            QMenuBar::item:selected { background-color: #505050; }
            QMenu { background-color: #3c3c3c; color: white; }
            QMenu::item:selected { background-color: #505050; }
            QStatusBar { color: white; }
        """

        # --- [추가] 메뉴바 생성 ---
        menu_bar = self.menuBar()

        # Preset 메뉴
        preset_menu = menu_bar.addMenu("Preset")
        # functools.partial을 사용하여 각 메뉴 액션에 다른 인자를 전달
        # 프리셋 버튼 추가
        from functools import partial
        sensitive_action = preset_menu.addAction("Sensitive")
        sensitive_action.triggered.connect(partial(self.apply_preset, "Sensitive"))
        balanced_action = preset_menu.addAction("Balanced")
        balanced_action.triggered.connect(partial(self.apply_preset, "Balanced"))
        stable_action = preset_menu.addAction("Stable")
        stable_action.triggered.connect(partial(self.apply_preset, "Stable"))
        # very_stable_action = preset_menu.addAction("Very Stable")
        # very_stable_action.triggered.connect(partial(self.apply_preset, "Very Stable"))

        # ViewMode 메뉴
        view_menu = menu_bar.addMenu("ViewMode")
        dark_mode_action = view_menu.addAction("Dark Mode")
        light_mode_action = view_menu.addAction("Light Mode")

        dark_mode_action.triggered.connect(self.set_dark_mode)
        light_mode_action.triggered.connect(self.set_light_mode)

        # --- [수정] 메인 레이아웃 구조 ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # 1. 상단 레이아웃 (하드웨어 정보, 프리셋 정보)
        top_layout = QHBoxLayout()
        self.hw_info_label = QLabel(f"CPU: {cpu_name}<br>GPU: {gpu_name}", self)
        top_layout.addWidget(self.hw_info_label)

        top_layout.addStretch(1)
        self.preset_info_label = QLabel("Preset: N/A", self)
        self.preset_info_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        top_layout.addWidget(self.preset_info_label)

        # 2. 중앙 레이아웃 (버튼 + 캠 화면 + 버튼)
        center_layout = QHBoxLayout()
        self.toggle_button = QPushButton("Input ON/OFF", self)
        self.rerecognize_button = QPushButton("Re-recognize\n Keyboard", self)
        # 이 숫자를 조절하여 원하는 너비로 맞출 수 있습니다.
        fixed_button_width = 150
        self.toggle_button.setFixedWidth(fixed_button_width)
        self.rerecognize_button.setFixedWidth(fixed_button_width)
        self.video_label = QLabel("Initializing...", self)
        self.video_label.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(self.toggle_button)
        center_layout.addWidget(self.video_label, 1)
        center_layout.addWidget(self.rerecognize_button)

        # 메인 레이아웃에 위젯과 레이아웃 추가
        self.main_layout.addLayout(top_layout)
        self.main_layout.addStretch(1) # 중앙 레이아웃 위쪽 빈 공간
        self.main_layout.addLayout(center_layout)
        self.main_layout.addStretch(1) # 중앙 레이아웃 아래쪽 빈 공간

        # --- 상태바 및 스레드 설정 (기존과 유사) ---
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Initializing hardware...")
        self.ref_keys_label = QLabel("Reference Keys: N/A", self)
        self.status_bar.addPermanentWidget(self.ref_keys_label)

        self.set_light_mode() # 기본 모드를 라이트 모드로 설정

        # 1. UI 상태 변수 초기화
        self.input_enabled = True

        # 2. 버튼 클릭과 슬롯 함수 연결
        self.toggle_button.clicked.connect(self.toggle_input)
        self.rerecognize_button.clicked.connect(self.force_rerecognize)

        # ▼▼▼ [수정] 카메라 자동 전환 로직 추가 ▼▼▼
        try:
            if USE_NETWORK_CAM:
                # 1. 네트워크 카메라를 우선 시도
                self.vs = VideoStream(src=NETWORK_CAM_URL)
            else:
                # 설정이 False면 처음부터 로컬 웹캠 시도
                self.vs = VideoStream(src=0)
        except IOError:
            # 2. 위에서 IOError 발생 시 (연결 실패 시) 여기로 넘어옴
            print("[WARN] Primary camera failed. Falling back to local webcam...")
            try:
                # 3. 로컬 웹캠(0번)으로 다시 연결 시도
                self.vs = VideoStream(src=0)
            except IOError as e:
                # 4. 모든 카메라 연결에 실패하면 에러 팝업창을 띄우고 종료
                error_title = "Camera Connection Failed"
                error_message = ("Could not connect to network or local webcam.\n\n"
                                "Please check your camera connection and restart the application.")
                QMessageBox.critical(self, error_title, error_message)
                sys.exit() # 사용자가 "OK"를 누르면 프로그램 종료
        self.thread = QThread()
        self.worker = Worker(self.vs)
        self.worker.moveToThread(self.thread)

        # 4. 모든 신호-슬롯 연결
        # Worker -> MainWindow
        self.worker.new_frame.connect(self.update_video_label)
        self.worker.update_status.connect(self.update_status_bar)
        self.worker.reference_keys_updated.connect(self.update_ref_keys_label)
        # MainWindow -> Worker
        self.toggle_input_signal.connect(self.worker.set_input_enabled)
        self.rerecognize_signal.connect(self.worker.force_rerecognize)
        self.apply_preset_signal.connect(self.worker.apply_preset)
        # ▼▼▼ 프리셋 정보 업데이트 신호-슬롯 연결 추가 ▼▼▼
        self.worker.preset_updated_signal.connect(self.update_preset_display)

        # 스레드 생명주기
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # 5. 스레드 시작 및 창 중앙 배치
        self.thread.start()
        self.center()
        
    def center(self):
        """애플리케이션 창을 화면 중앙에 배치합니다."""
        screen_geometry = self.screen().geometry()
        window_geometry = self.frameGeometry()
        window_geometry.moveCenter(screen_geometry.center())
        self.move(window_geometry.topLeft())

    # ▼▼▼ [추가] 토글 버튼 클릭 시 실행될 슬롯 ▼▼▼
    def toggle_input(self):
        # 1. UI의 상태 변수를 반전시킴
        self.input_enabled = not self.input_enabled

        # 2. 버튼 텍스트를 현재 상태에 맞게 변경
        if self.input_enabled:
            self.toggle_button.setText("Input: ON")
        else:
            self.toggle_button.setText("Input: OFF")

        # 3. Worker 스레드에게 변경된 상태를 신호로 보냄
        self.toggle_input_signal.emit(self.input_enabled)

    def force_rerecognize(self):
        """재인식 버튼 클릭 시 Worker에게 신호를 보냅니다."""
        self.rerecognize_signal.emit()

    def apply_preset(self, preset_name):
        """프리셋 메뉴 선택 시 Worker에게 프리셋 이름을 신호로 보냅니다."""
        self.apply_preset_signal.emit(preset_name)
        self.status_bar.showMessage(f"Preset changed to: {preset_name}", 2000)

    # ▼▼▼ 프리셋 정보 표시를 위한 새로운 슬롯 추가 ▼▼▼
    def update_preset_display(self, name, on, off, ema):
        text = f"Preset: {name}\nON:{on:.2f} OFF:{off:.2f} EMA:{ema:.2f}"
        self.preset_info_label.setText(text)

    def update_video_label(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)

    def update_status_bar(self, full_status_text):
        self.status_bar.showMessage(full_status_text)

    def update_ref_keys_label(self, text):
        self.ref_keys_label.setText(text)

    # MainWindow 클래스 내부 (예: update_ref_keys_label 메서드 다음)
    def update_preset_display(self, text):
        """Worker로부터 받은 프리셋 정보로 라벨을 업데이트합니다."""
        self.preset_info_label.setText(text)

    # --- [추가] 다크/라이트 모드 전환을 위한 슬롯 ---
    def set_dark_mode(self):
        self.setStyleSheet(self.dark_mode_style)

    def set_light_mode(self):
        self.setStyleSheet(self.light_mode_style)

    def closeEvent(self, event):
        print("Stopping threads...")
        self.worker.stop()
        self.vs.stop()
        event.accept()

# ===== 프로그램 실행부 =====
# if __name__ == "__main__":
#    pyautogui.FAILSAFE = False
#    app = QApplication(sys.argv)
#    window = MainWindow()
#    window.show()
#    sys.exit(app.exec())

# ===== 프로그램 실행부 (스플래시 스크린 적용) =====
if __name__ == "__main__":
    pyautogui.FAILSAFE = False
    app = QApplication(sys.argv)

    # 1. 스플래시 스크린에 사용할 이미지(Pixmap) 생성
    splash_pix = QPixmap("logo.png")

    # ▼▼▼ [수정] 원하는 크기(예: 너비 000px)로 조정합니다 ▼▼▼
    # 높이는 비율에 맞게 자동 조절됩니다. 숫자를 바꿔서 크기를 조절하세요.
    scaled_pix = splash_pix.scaledToWidth(300, Qt.SmoothTransformation)

    # 조정된 이미지(scaled_pix)로 스플래시 스크린 생성
    splash = QSplashScreen(scaled_pix, Qt.WindowStaysOnTopHint)

    # ... (이하 스플래시 스크린 로직은 동일) ...
    splash.show()
    splash.showMessage("Loading AI models...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
    window = MainWindow()
    splash.finish(window)
    window.show()

    sys.exit(app.exec())