import os
import whisper

# 音声ファイルが格納されているディレクトリのパスを取得
audio_folder = os.path.join(os.path.dirname(__file__), "transcriptAudio")

# Whisperモデルをロード
model = whisper.load_model("large", device="cuda")

# テキスト出力ディレクトリのパスを設定
text_output_dir = os.path.join(os.path.dirname(__file__), "textOutput")

if not os.path.exists(text_output_dir):
    os.makedirs(text_output_dir)

# 音声ファイルを処理
for audio_filename in os.listdir(audio_folder):
    if audio_filename.endswith((".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac")):
        audio_path = os.path.join(audio_folder, audio_filename)
        
        # 音声ファイルを文字起こし
        result = model.transcribe(audio_path)
        
        # テキストファイルのパスを設定
        text_filename = os.path.splitext(audio_filename)[0] + ".txt"
        output_text_path = os.path.join(text_output_dir, text_filename)
        
        # テキストファイルに書き込み
        with open(output_text_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

