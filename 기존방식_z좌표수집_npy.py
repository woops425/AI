import pandas as pd
import numpy as np
import glob

# --- 설정 --- #
CSV_PATH = "data/finger_data_*.csv"
OUTPUT_X = "X_zonly_sequences.npy"
OUTPUT_Y = "y_zonly_labels.npy"
WINDOW_SIZE = 30
APPLY_LABEL_EXTENSION = False  # 라벨 확장 기능 켜기 (+1 프레임)

# --- 1. CSV 읽기 --- #
csv_files = glob.glob(CSV_PATH)
print(f"CSV 파일 수: {len(csv_files)}개")

df_list = []
for file in csv_files:
    df = pd.read_csv(file)
    df = df.fillna(0.0)

    # --- (선택) +1 프레임 확장 라벨링 --- #
    if APPLY_LABEL_EXTENSION:
        pressed_array = df["pressed"].values.copy()
        for idx in np.where(pressed_array == 1)[0]:
            for delta in [1]:
                neighbor = idx + delta
                if 0 <= neighbor < len(pressed_array):
                    pressed_array[neighbor] = 1
        df["pressed"] = pressed_array

    df_list.append(df)

# --- 2. 데이터 통합 --- #
all_data = pd.concat(df_list, ignore_index=True)
print(f"전체 데이터 크기: {all_data.shape}")

# --- 3. 시퀀스 생성 --- #
def create_z_sequences(df, window_size=30):
    X, y = [], []
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i+window_size]
        z_only = window[['index_z']].values  # (30, 1)
        label = 1 if (window['pressed'] == 1).any() else 0
        X.append(z_only)
        y.append(label)
    return np.array(X), np.array(y)

X, y = create_z_sequences(all_data, window_size=WINDOW_SIZE)
print(f"시퀀스 형태: X={X.shape}, y={y.shape}")

# --- 4. 정규화 --- #
X = np.nan_to_num(X)
max_val = np.max(X)
if max_val == 0: max_val = 1
X = X / max_val

# --- 5. 셔플 --- #
indices = np.arange(len(X))
np.random.seed(42)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# --- 6. 저장 --- #
np.save(OUTPUT_X, X)
np.save(OUTPUT_Y, y)
print(f"저장 완료 → {OUTPUT_X}, {OUTPUT_Y}")