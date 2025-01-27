import os
from moviepy.editor import VideoFileClip
import whisper
from dotenv import load_dotenv
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import librosa


load_dotenv()

model_id = "kotoba-tech/kotoba-whisper-v2.0"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)

# Set the paths
video_path = os.getenv("VIDEO_FILE_PATH")

audio_folder = os.path.join(os.path.dirname(__file__), "tmpAudio")
output_audio_path = os.path.join(audio_folder, "temp_audio.mp3")

# Extract audio from the video
video = VideoFileClip(video_path)
video.audio.write_audiofile(output_audio_path)

# 音声ファイルの読み込み
audio_input, sample_rate = sf.read(output_audio_path)
if sample_rate != 16000:
    audio_input = librosa.resample(audio_input, orig_sr=sample_rate, target_sr=16000)
    sample_rate = 16000

# 音声データの前処理
input_features = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)

# 推論の実行
generated_ids = model.generate(input_features)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

text_output_dir = os.path.join(os.path.dirname(__file__), "textOutput")

if not os.path.exists(text_output_dir):
    os.makedirs(text_output_dir)

video_filename = os.path.basename(video_path)
text_filename = os.path.splitext(video_filename)[0] + ".txt"
output_text_path = os.path.join(text_output_dir, text_filename)

with open(output_text_path, "w", encoding="utf-8") as f:
    f.write(transcription["text"])

# Remove the temporary audio file
os.remove(output_audio_path)

