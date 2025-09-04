import torch
import utils
import soundfile as sf
from models import SynthesizerTrn
from text import text_to_sequence, symbols  # 音素转 id

# ===== 配置 =====
checkpoint_path = "./logs/mxj_model/G_14000.pth"
config_path = "./dataset/mxj_config.json"
output_wav = "./dataset/test_14000.wav"
phoneme_text = "xə↑ tʰa→ ts⁼aɪ↓ i↓tʃʰi↓↑ ni↓↑ ..."

# ===== 读取配置 =====
hps = utils.get_hparams_from_file(config_path)

# 单说话人模型，添加 symbols
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

# ===== 音素文本转序列（不使用 cleaner） =====
sequence = text_to_sequence(phoneme_text, hps.symbols, cleaner_names=[])
x_tst = torch.LongTensor(sequence).unsqueeze(0)

# ===== 推理 =====
with torch.no_grad():
    audio = net_g.infer(x_tst, torch.LongTensor([len(sequence)]))[0][0,0].cpu().numpy()

# ===== 保存 wav =====
sf.write(output_wav, audio, hps.data.sampling_rate)
print(f"合成完成，输出文件: {output_wav}")
