import os
import soundfile as sf
import numpy as np

DATA_DIR = "dataset/audio"
TARGET_DTYPE = np.float32  # numpy dtype
TARGET_SUBTYPE = 'FLOAT'   # 对应 soundfile subtype

def process_wav(path, target_dtype=TARGET_DTYPE):
    data, sr = sf.read(path)
    changed = False

    # 如果是双声道，取平均
    if data.ndim == 2:
        data = data.mean(axis=1)
        changed = True

    # 统一采样大小
    if data.dtype != TARGET_DTYPE:
        data = data.astype(TARGET_DTYPE)
        changed = True

    if changed:
        sf.write(path, data, sr, subtype=TARGET_SUBTYPE)
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
