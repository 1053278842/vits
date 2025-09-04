import os
import soundfile as sf
import numpy as np

DATA_DIR = "dataset/audio"
TARGET_DTYPE = 'float32'  # 统一为 float32

def process_wav(path, target_dtype=TARGET_DTYPE):
    data, sr = sf.read(path)
    changed = False

    # 如果是双声道，取平均
    if data.ndim == 2:
        data = data.mean(axis=1)
        changed = True

    # 如果采样大小不一致，统一为 float32
    if data.dtype != np.dtype(target_dtype):
        data = data.astype(target_dtype)
        changed = True

    if changed:
        sf.write(path, data, sr, subtype=target_dtype)
        return True
    return False

def main():
    total_count = 0
    fixed_count = 0

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(".wav"):
                total_count += 1
                path = os.path.join(root, file)
                if process_wav(path):
                    fixed_count += 1

    print(f"总文件数: {total_count}")
    print(f"已处理并覆盖: {fixed_count}")

if __name__ == "__main__":
    main()
