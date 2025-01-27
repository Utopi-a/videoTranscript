import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from scipy import signal

# 設定
model_id = "kotoba-tech/kotoba-whisper-v2.0"
device = "cuda" if torch.cuda.is_available() else "cpu"

# プロセッサとモデルのロード
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)

import soundfile as sf

# 音声ファイルの読み込み
audio_data, sample_rate = sf.read("tmpAudio/temp_audio.mp3")

print(sample_rate)

if sample_rate != 16000:
    new_sample_rate = 16000
    print("resample")
    number_of_samples = round(len(audio_data) * float(new_sample_rate) / sample_rate)
    audio_data = signal.resample(audio_data, number_of_samples)
    print("end_resample")

# 音声データの前処理
input_features = processor(audio_data, sampling_rate=new_sample_rate, return_tensors="pt").input_features.to(device)

# 推論の実行
generated_ids = model.generate(input_features)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(transcription)