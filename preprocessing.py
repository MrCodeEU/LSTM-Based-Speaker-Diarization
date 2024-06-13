import glob
from multiprocessing import Pool

import librosa
import numpy as np
import webrtcvad
import os
import xml.etree.ElementTree as ET

from tqdm import tqdm

from extract_features import extract_features


def load_segments(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    segments = []
    for segment in root.findall('segment'):
        start_time = float(segment.attrib['starttime'])
        end_time = float(segment.attrib['endtime'])
        participant = segment.attrib['participant']
        segments.append((start_time, end_time, participant))

    return segments


def apply_vad(segments, sr, vad_aggressiveness=3):
    vad = webrtcvad.Vad(vad_aggressiveness)

    speech_segments = []
    for segment in segments:
        start_time, audio = segment
        frame_duration = 30  # Duration of each frame in ms
        frame_samples = int(frame_duration * sr / 1000)

        frames = []
        for i in range(0, len(audio), frame_samples):
            frame = audio[i:i + frame_samples]
            if len(frame) < frame_samples:
                frame = np.pad(frame, (0, frame_samples - len(frame)))
            frames.append(frame)

        speech_frames = []
        for frame in frames:
            # convert audio to 16-bit PCM format for vad
            frame_16 = np.int16(frame * 32768)
            is_speech = vad.is_speech(frame_16.tobytes(), sample_rate=sr)
            if is_speech:
                speech_frames.append(frame)

        if speech_frames:
            speech_segment = np.concatenate(speech_frames)
            speech_segments.append((start_time, speech_segment))

    return speech_segments


def segment_audio(audio, sr, segment_length=30, overlap=10):
    # Calculate the number of samples per segment and overlap
    segment_samples = int(segment_length * sr / 1000)
    overlap_samples = int(overlap * sr / 1000)

    # Segment the audio
    segments = []
    for i in range(0, len(audio) - segment_samples + 1, segment_samples - overlap_samples):
        segment = audio[i:i + segment_samples]
        segments.append((i / sr, segment))

    return segments, sr


def load_audio_and_xml_segments(train_dir, segments_dir, meeting_dir):
    """Loads audio and XML segment data_icsi for a given meeting."""
    audio_path = os.path.join(train_dir, meeting_dir, f"{meeting_dir}.interaction.wav")
    audio, _ = librosa.load(audio_path, sr=None)  # Load audio (ignore sample rate for now)
    audio_segments, sr = segment_audio(audio_path)
    xml_files = glob.glob(os.path.join(segments_dir, f"{meeting_dir}.*.segs.xml"))
    xml_segments = []
    for xml_file in xml_files:
        xml_segments.extend(load_segments(xml_file))
    xml_segments.sort(key=lambda x: x[0])  # Sort by start time
    return audio, audio_segments, xml_segments, sr


def process_segment(args):
    """Extracts log-mel spectrograms for a single audio segment and its speaker label."""
    participant, (start_time, audio_segment), sr = args
    log_mel_spectrogram = extract_features(audio_segment, sr)
    return log_mel_spectrogram, participant


def extract_features_parallel(speech_segments, sr):
    """Extracts log-mel spectrograms in parallel using multiprocessing."""
    args = [(p, seg, sr) for p, segments in speech_segments.items() for seg in segments]
    with Pool() as p:
        results = list(tqdm(p.imap(process_segment, args), total=len(args), desc="Extracting features"))
    log_mel_spectrograms, labels = zip(*results)
    return np.array(log_mel_spectrograms), list(labels)