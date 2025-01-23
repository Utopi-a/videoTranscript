import os
from moviepy.editor import VideoFileClip
import whisper
from dotenv import load_dotenv

load_dotenv()

# Set the paths
video_path = os.getenv("VIDEO_FILE_PATH")

audio_folder = os.path.join(os.path.dirname(__file__), "tmpAudio")
output_audio_path = os.path.join(audio_folder, "temp_audio.mp3")

# Extract audio from the video
video = VideoFileClip(video_path)
video.audio.write_audiofile(output_audio_path)

# Load the Whisper ASR model
model = whisper.load_model("small")

# Transcribe the extracted audio
result = model.transcribe(output_audio_path)

text_output_dir = os.path.join(os.path.dirname(__file__), "textOutput")

if not os.path.exists(text_output_dir):
    os.makedirs(text_output_dir)

video_filename = os.path.basename(video_path)
text_filename = os.path.splitext(video_filename)[0] + ".txt"
output_text_path = os.path.join(text_output_dir, text_filename)

with open(output_text_path, "w", encoding="utf-8") as f:
    f.write(result["text"])

# Remove the temporary audio file
os.remove(output_audio_path)

