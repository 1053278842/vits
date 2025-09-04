import torch
import utils
import soundfile as sf
from models import SynthesizerTrn
from text import symbols  # 直接用 symbols，不用 text_to_sequence

# ===== 配置 =====
checkpoint_path = "./logs/mxj_model/G_14000.pth"
config_path = "./dataset/mxj_config.json"
output_wav = "./dataset/test_14000.wav"
with open("dataset/filelist_mxj.txt.cleaned_val", "r", encoding="utf-8") as f:
    line = f.readline()
phoneme_text = line.strip().split("|")[1]  # 第二列是音素


# ===== 读取配置 =====
hps = utils.get_hparams_from_file(config_path)
hps.symbols = symbols

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

# ===== 音素文本转 id（手动映射） =====
sequence = [hps.symbols.index(s) for s in phoneme_text.split() if s in hps.symbols]
x_tst = torch.LongTensor(sequence).unsqueeze(0)

# ===== 推理 =====
with torch.no_grad():
    audio = net_g.infer(x_tst, torch.LongTensor([len(sequence)]))[0][0,0].cpu().numpy()

# ===== 保存 wav =====
sf.write(output_wav, audio, hps.data.sampling_rate)
print(f"合成完成，输出文件: {output_wav}")
