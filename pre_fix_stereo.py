import os
import soundfile as sf
import numpy as np

DATA_DIR = "dataset/audio"

def process_wav(path):
    data, sr = sf.read(path)
    if data.ndim == 2:  # 双声道
        # 取左右声道平均
        data = data.mean(axis=1)
        sf.write(path, data, sr)
        return True
    return False

def main():
    stereo_count = 0
    fixed_count = 0
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(".wav"):
                path = os.path.join(root, file)
                data, sr = sf.read(path)
                if data.ndim == 2:
                    stereo_count += 1
                    if process_wav(path):
                        fixed_count += 1
    print(f"发现双声道文件: {stereo_count}")
    print(f"已处理并覆盖: {fixed_count}")

if __name__ == "__main__":
    main()
