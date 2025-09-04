import os
import librosa

DATA_DIR = "./wavs"  # 改成你音频文件的目录
MIN_LENGTH = 1024

min_len = float("inf")
short_count = 0
stereo_files = []

for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.lower().endswith(".wav"):
            filepath = os.path.join(root, file)
            # sr=None 表示不重采样，返回原始声道
            wav, sr = librosa.load(filepath, sr=None, mono=False)

            # 检查声道数
            if wav.ndim > 1:
                stereo_files.append(filepath)

            # 统一转成单声道再检查长度
            if wav.ndim > 1:
                wav = wav.mean(axis=0)

            if len(wav) < min_len:
                min_len = len(wav)
            if len(wav) < MIN_LENGTH:
                short_count += 1

print(f"最短音频长度: {min_len} 采样点")
print(f"小于 {MIN_LENGTH} 的音频文件数量: {short_count}")
print(f"双声道音频数量: {len(stereo_files)}")

if stereo_files:
    print("双声道文件示例（最多列出前10个）：")
    for f in stereo_files[:10]:
        print("  ", f)
