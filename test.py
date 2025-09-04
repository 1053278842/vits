import torch
import utils
import soundfile as sf
import random
from models import SynthesizerTrn
from text import text_to_sequence
import commons

# ===== 配置 =====
checkpoint_path = "./logs/mxj_model/G_14000.pth"
config_path = "./dataset/mxj_config.json"
val_filelist = "./dataset/filelist_mxj.txt.cleaned_val"
output_wav = "./dataset/test_14000.wav"

# ===== 读取配置 =====
hps = utils.get_hparams_from_file(config_path)

# ===== 构建模型 =====
net_g = SynthesizerTrn(
    len(hps.symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=0,
    **hps.model
)
_ = net_g.eval()
_ = utils.load_checkpoint(checkpoint_path, net_g, None)

# ===== 从 val 里随机选一句 =====
with open(val_filelist, "r", encoding="utf-8") as f:
    lines = f.readlines()
line = random.choice(lines).strip()
phoneme_text = line.split("|")[1]  # 第二列是音素

# ===== 音素转序列 =====
sequence = text_to_sequence(phoneme_text, hps.data.text_cleaners)
if hps.data.add_blank:
    sequence = commons.intersperse(sequence, 0)
x_tst = torch.LongTensor(sequence).unsqueeze(0)
x_lengths = torch.LongTensor([x_tst.size(1)])

# ===== 推理 =====
with torch.no_grad():
    audio = net_g.infer(x_tst, x_lengths)[0][0,0].cpu().numpy()

# ===== 保存 wav =====
sf.write(output_wav, audio, hps.data.sampling_rate)
print(f"合成完成: {output_wav}")
