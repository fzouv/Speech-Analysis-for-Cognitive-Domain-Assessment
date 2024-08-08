import os
from pydub import AudioSegment
import webrtcvad
import numpy as np
import wave
import contextlib


def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1, "Audio file should be mono"
        sample_width = wf.getsampwidth()
        assert sample_width == 2, "Audio file should be 16-bit"
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000), "Sample rate should be 8000, 16000, 32000, or 48000 Hz"
        pcm_data = wf.readframes(wf.getnframes())
    return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    while offset + n < len(audio):
        yield audio[offset:offset + n]
        offset += n


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = []
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame, sample_rate)

        if not triggered:
            ring_buffer.append(frame)
            if len(ring_buffer) > num_padding_frames:
                ring_buffer.pop(0)
            num_voiced = len([f for f in ring_buffer if vad.is_speech(f, sample_rate)])
            if num_voiced > 0.9 * num_padding_frames:
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer = []
        else:
            voiced_frames.append(frame)
            num_unvoiced = len([f for f in voiced_frames[-num_padding_frames:] if not vad.is_speech(f, sample_rate)])
            if num_unvoiced > 0.9 * num_padding_frames:
                triggered = False
                yield b''.join(voiced_frames)
                ring_buffer = []
                voiced_frames = []

    if voiced_frames:
        yield b''.join(voiced_frames)


def process_audio(file_path, output_dir, interviewee_id):
    audio, sample_rate = read_wave(file_path)
    vad = webrtcvad.Vad(0)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)

    for i, segment in enumerate(segments):
        output_path = os.path.join(output_dir, f'{interviewee_id}_segment_{i}.wav')
        write_wave(output_path, segment, sample_rate)

if __name__ == "__main__":
    input_dir = '/Users/fotianazouvani/Downloads/SLP/Diss/Control_converted/preprocessed'
    output_dir = '/Users/fotianazouvani/Downloads/SLP/Diss/vad_control'
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.wav'):
            interviewee_id = os.path.splitext(file_name)[0]  # Extract interviewee ID from the file name
            file_path = os.path.join(input_dir, file_name)
            process_audio(file_path, output_dir, interviewee_id)
