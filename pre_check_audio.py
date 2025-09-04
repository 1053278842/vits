import os
import torchaudio

# 你的数据路径
DATA_DIR = "dataset/audio/mxj"
# 频谱参数，和 hps 里保持一致
FILTER_LENGTH = 1024  # 例如 1024 或 2048

def check_audio_files(data_dir, filter_length):
    min_len = 1e9
    bad_files = []

    for root, _, files in os.walk(data_dir):
        for f in files:
            if not f.endswith(".wav"):
                continue
            path = os.path.join(root, f)
            try:
                wav, sr = torchaudio.load(path)
                length = wav.size(-1)
                if length < min_len:
                    min_len = length
                if length < filter_length:
                    bad_files.append((path, length))
            except Exception as e:
                print(f"[ERROR] {path}: {e}")

    print(f"\n最短音频长度: {min_len} 采样点")
    print(f"小于 {filter_length} 的音频文件数量: {len(bad_files)}\n")
    for path, length in bad_files[:30]:  # 只先打印前 30 个
        print(f"{path} -> {length}")
    if len(bad_files) > 30:
        print(f"... 还有 {len(bad_files) - 30} 个")

if __name__ == "__main__":
    check_audio_files(DATA_DIR, FILTER_LENGTH)
