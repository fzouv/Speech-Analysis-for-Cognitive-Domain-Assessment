import os
from pydub import AudioSegment

def preprocess_audio(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)  # Mono
    audio = audio.set_frame_rate(16000)  # Set sample rate to 16000 Hz
    audio = audio.set_sample_width(2)  # 16-bit PCM
    audio.export(output_path, format="wav")

def preprocess_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.wav'):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            preprocess_audio(input_path, output_path)
            print(f"Processed {input_path} -> {output_path}")

if __name__ == "__main__":
    input_dir = '/Users/fotianazouvani/Downloads/SLP/Diss/MCI_converted'
    output_dir = '/Users/fotianazouvani/Downloads/SLP/Diss/MCI_converted/preprocessed'
    preprocess_directory(input_dir, output_dir)
